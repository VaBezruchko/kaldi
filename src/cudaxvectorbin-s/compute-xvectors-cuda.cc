// cudafeatbin/compute-mfcc-feats-cuda.cc
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Justin Luitjens
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
#include "cudamatrix/cu-allocator.h"

#include "feat/wave-reader.h"

#include "cudafeat/feature-spectral-cuda.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-allocator.h"

#include "ivector/voice-activity-detection.h"
#include "feat/feature-functions.h"

using namespace kaldi;
using namespace kaldi::nnet3;

bool ComputeMFCC(WaveData *wave_data, BaseFloat min_duration, int32 channel,
		CudaSpectralFeatures &mfcc, Matrix<BaseFloat> *features,
		std::string &utt) {

	if (wave_data->Duration() < min_duration) {
		KALDI_WARN << "File: " << utt << " is too short ("
				<< wave_data->Duration() << " sec): producing no output.";
		return false;
	}
	int32 num_chan = wave_data->Data().NumRows(), this_chan = channel;
	{  // This block works out the channel (0=left, 1=right...)
		KALDI_ASSERT(num_chan > 0);  // should have been caught in
		// reading code if no channels.
		if (channel == -1) {
			this_chan = 0;
			if (num_chan != 1)
				KALDI_WARN << "Channel not specified but you have data with "
						<< num_chan << " channels; defaulting to zero";
		} else {
			if (this_chan >= num_chan) {
				KALDI_WARN << "File with id " << utt << " has " << num_chan
						<< " channels but you specified channel " << channel
						<< ", producing no output.";
				return false;
			}
		}
	}

	SubVector<BaseFloat> waveform(wave_data->Data(), this_chan);

	CuVector<BaseFloat> cu_waveform(waveform);
	CuMatrix<BaseFloat> cu_features;
	mfcc.ComputeFeatures(cu_waveform, wave_data->SampFreq(), 1.0, &cu_features);
	features->Resize(cu_features.NumRows(), cu_features.NumCols());
	features->CopyFromMat(cu_features);

	return true;

}

void Nnet3XvectorInit(ParseOptions &po, NnetSimpleComputationOptions *opts,
		CachingOptimizingCompilerOptions *compiler_config,
		std::string *cached_compiler_out, std::string *cached_compiler_in,
		int32 *chunk_size, int32 *min_chunk_size, bool *pad_input) {

	opts->acoustic_scale = 1.0; // by default do no scaling in this recipe.
	opts->Register(&po);

	compiler_config->Register(&po);

	po.Register("chunk-size", chunk_size,
			"If set, extracts xectors from specified chunk-size, and averages.  "
					"If not set, extracts an xvector from all available features.");
	po.Register("min-chunk-size", min_chunk_size,
			"Minimum chunk-size allowed when extracting xvectors.");
	po.Register("pad-input", pad_input,
			"If true, duplicate the first and "
					"last frames of the input features as required to equal min-chunk-size.");
	po.Register("cached-compiler-in", cached_compiler_in,
			"If set, read the cached compiler from the specified file path.");
	po.Register("cached-compiler-out", cached_compiler_out,
			"If set, write the cached compiler to the specified file path.");

}

// Computes an xvector from a chunk of speech features.
static void RunNnetComputation(const MatrixBase<BaseFloat> &features,
		const Nnet &nnet, CachingOptimizingCompiler *compiler,
		Vector<BaseFloat> *xvector) {
	ComputationRequest request;
	request.need_model_derivative = false;
	request.store_component_stats = false;
	request.inputs.push_back(IoSpecification("input", 0, features.NumRows()));
	IoSpecification output_spec;
	output_spec.name = "output";
	output_spec.has_deriv = false;
	output_spec.indexes.resize(1);
	request.outputs.resize(1);
	request.outputs[0].Swap(&output_spec);
	std::shared_ptr<const NnetComputation> computation(
			compiler->Compile(request));
	Nnet *nnet_to_update = NULL;  // we're not doing any update.
	NnetComputer computer(NnetComputeOptions(), *computation, nnet,
			nnet_to_update);
	CuMatrix<BaseFloat> input_feats_cu(features);
	computer.AcceptInput("input", &input_feats_cu);
	computer.Run();
	CuMatrix<BaseFloat> cu_output;
	computer.GetOutputDestructive("output", &cu_output);
	xvector->Resize(cu_output.NumCols());
	xvector->CopyFromVec(cu_output.Row(0));
}

bool Nnet3XvectorCompute(const MatrixBase<BaseFloat> &features, std::string &utt,
		Vector<BaseFloat> *xvector_avg, const Nnet &nnet, int32 chunk_size,
		int32 min_chunk_size, bool pad_input,
		CachingOptimizingCompiler *compiler) {

	if (features.NumRows() == 0) {
		KALDI_WARN << "Zero-length utterance: " << utt;
		return false;
	}

	int32 num_rows = features.NumRows(), feat_dim = features.NumCols(),
			this_chunk_size = chunk_size;
	if (!pad_input && num_rows < min_chunk_size) {
		KALDI_WARN << "Minimum chunk size of " << min_chunk_size
				<< " is greater than the number of rows " << "in utterance: "
				<< utt;
		return false;
	} else if (num_rows < chunk_size) {
//		KALDI_LOG << "Chunk size of " << chunk_size << " is greater than "
//				<< "the number of rows in utterance: " << utt
//				<< ", using chunk size  of " << num_rows;
		this_chunk_size = num_rows;
	} else if (chunk_size == -1) {
		this_chunk_size = num_rows;
	}

	int32 num_chunks = ceil(num_rows / static_cast<BaseFloat>(this_chunk_size));

	BaseFloat tot_weight = 0.0;

	// Iterate over the feature chunks.
	for (int32 chunk_indx = 0; chunk_indx < num_chunks; chunk_indx++) {
		// If we're nearing the end of the input, we may need to shift the
		// offset back so that we can get this_chunk_size frames of input to
		// the nnet.
		int32 offset = std::min(this_chunk_size,
				num_rows - chunk_indx * this_chunk_size);
		if (!pad_input && offset < min_chunk_size)
			continue;
		SubMatrix<BaseFloat> sub_features(features,
				chunk_indx * this_chunk_size, offset, 0, feat_dim);
		Vector<BaseFloat> xvector;
		tot_weight += offset;

		// Pad input if the offset is less than the minimum chunk size
		if (pad_input && offset < min_chunk_size) {
			Matrix<BaseFloat> padded_features(min_chunk_size, feat_dim);
			int32 left_context = (min_chunk_size - offset) / 2;
			int32 right_context = min_chunk_size - offset - left_context;
			for (int32 i = 0; i < left_context; i++) {
				padded_features.Row(i).CopyFromVec(sub_features.Row(0));
			}
			for (int32 i = 0; i < right_context; i++) {
				padded_features.Row(min_chunk_size - i - 1).CopyFromVec(
						sub_features.Row(offset - 1));
			}
			padded_features.Range(left_context, offset, 0, feat_dim).CopyFromMat(
					sub_features);
			RunNnetComputation(padded_features, nnet, compiler, &xvector);
		} else {
			RunNnetComputation(sub_features, nnet, compiler, &xvector);
		}
		xvector_avg->AddVec(offset, xvector);
	}
	xvector_avg->Scale(1.0 / tot_weight);
	return true;

}

int main(int argc, char *argv[]) {
	try {

		const char *usage =
				"Create xvectors files.\n"
						"Usage:  compute-xvectors-cuda [options...] <wav-rspecifier> <xvectors-wspecifier>\n";

		// construct all the global objects
		ParseOptions po(usage);
		Timer timer;

		//------------------------  MFCC options
		MfccOptions mfcc_opts;

		int32 channel = -1;
		BaseFloat min_duration = 0.0;

		// Register the MFCC option struct
		mfcc_opts.Register(&po);

		po.Register("channel", &channel,
				"Channel to extract (-1 -> expect mono, "
						"0 -> left, 1 -> right)");
		po.Register("min-duration", &min_duration,
				"Minimum duration of segments "
						"to process (in seconds).");

		//------------------------  VAD options

		VadEnergyOptions vad_opts;
		// Register the VAD option struct
		vad_opts.Register(&po);

		//------------------------  SlidingWindowCmn options

		SlidingWindowCmnOptions swc_opts;

		// Register the SlidingWindowCmn option struct
		swc_opts.Register(&po);

		//------------------------  NNET3 options

		NnetSimpleComputationOptions nnet3_opts;
		CachingOptimizingCompilerOptions compiler_config;

		std::string cached_compiler_out;
		std::string cached_compiler_in;
		int32 chunk_size = -1, min_chunk_size = 100;
		bool pad_input = true;
		Nnet nnet;

		Nnet3XvectorInit(po, &nnet3_opts, &compiler_config,
				&cached_compiler_out, &cached_compiler_in, &chunk_size,
				&min_chunk_size, &pad_input);

		//-----------------------

		RegisterCuAllocatorOptions(&po);
		CuDevice::RegisterDeviceOptions(&po);

		po.Read(argc, argv);
		if (po.NumArgs() != 3) {
			po.PrintUsage();
			exit(1);
		}

		CuDevice::Instantiate().SelectGpuId("yes");
		CuDevice::Instantiate().AllowMultithreading();


		std::string nnet_rxfilename = po.GetArg(1);
		std::string wav_rspecifier = po.GetArg(2);
		std::string vector_wspecifier = po.GetArg(3);


		//------------------------  NNET3 options cache

		ReadKaldiObject(nnet_rxfilename, &nnet);
		SetBatchnormTestMode(true, &nnet);
		SetDropoutTestMode(true, &nnet);
		CollapseModel(CollapseModelConfig(), &nnet);

		CachingOptimizingCompiler compiler(nnet, nnet3_opts.optimize_config,
				compiler_config);

		if (!cached_compiler_in.empty()) {
			KALDI_LOG << "Reading cache from " << cached_compiler_in;
			bool cache_binary_in;
			Input ki(cached_compiler_in, &cache_binary_in);
			compiler.ReadCache(ki.Stream(), cache_binary_in);
		}

		//-----------------------

		CudaSpectralFeatures mfcc(mfcc_opts);

		SequentialTableReader<WaveHolder> reader(wav_rspecifier);
		BaseFloatVectorWriter vector_writer(vector_wspecifier);

		int32 num_utts = 0, num_success = 0, num_fail = 0;
		int64 frame_count = 0;
		int32 xvector_dim = nnet.OutputDim("output");

		for (; !reader.Done(); reader.Next()) {
			num_utts++;

			Matrix<BaseFloat> features;

			std::string utt = reader.Key();
			WaveData &wave_data = reader.Value();

			if (ComputeMFCC(&wave_data, min_duration, channel, mfcc, &features,
					utt) == false) {
				num_fail++;
				continue;
			}


			//VAD
			Vector<BaseFloat> vad_result(features.NumRows());
			ComputeVadEnergy(vad_opts, features, &vad_result);

			//apply cmvn sliding
			Matrix<BaseFloat> cmvn_feat(features.NumRows(), features.NumCols(),
					kUndefined);
			SlidingWindowCmn(swc_opts, features, &cmvn_feat);

			//select voiced frames
			int32 dim = 0;
			for (int32 i = 0; i < vad_result.Dim(); i++) {
				if (vad_result(i) != 0.0) {
					dim++;
				}
			}



			Matrix<BaseFloat> voiced_features(dim, features.NumCols());
			int32 index = 0;
			for (int32 i = 0; i < features.NumRows(); i++) {
				if (vad_result(i) != 0.0) {
					KALDI_ASSERT(vad_result(i) == 1.0); // should be zero or one.
					voiced_features.Row(index).CopyFromVec(cmvn_feat.Row(i));
					index++;
				}
			}
			KALDI_ASSERT(index == dim);


			//remove tail over chunk_size for solve performance penalty
			MatrixBase<BaseFloat> *norm_features = &voiced_features;
			if (chunk_size != -1 && voiced_features.NumRows() > chunk_size) {
				int32 blns = voiced_features.NumRows() % chunk_size;
				if (blns != 0) {
					SubMatrix<BaseFloat> sub_features(voiced_features,
									0, voiced_features.NumRows() - blns, 0, features.NumCols());
					norm_features = &sub_features;
				}
			}

			//xvector compute
			Vector<BaseFloat> xvector(xvector_dim, kSetZero);
			if (Nnet3XvectorCompute(*norm_features, utt, &xvector, nnet,
					chunk_size, min_chunk_size, pad_input, &compiler)
					== false) {
				num_fail++;
				continue;
			}

			vector_writer.Write(utt, xvector);

			if (num_utts % 100 == 0) {
				KALDI_LOG << "Processed " << num_utts << " utterances";
			}
			KALDI_VLOG(2) << "Processed features for key " << utt;
			frame_count += features.NumRows();
			num_success++;
		}

		CuDevice::Instantiate().PrintProfile();

		double elapsed = timer.Elapsed();
		KALDI_LOG << "Time taken " << elapsed
				<< "s: real-time factor assuming 100 frames/sec is "
				<< (elapsed * 100.0 / frame_count);
		KALDI_LOG << "Done " << num_success << " utterances, failed for "
				<< num_fail;

		if (!cached_compiler_out.empty()) {
			KALDI_LOG << "Writing cache to " << cached_compiler_out;
			bool binary_write = true;
			Output ko(cached_compiler_out, binary_write);
			compiler.WriteCache(ko.Stream(), binary_write);
		}

		return (num_success != 0 ? 0 : 1);
	} catch (const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}

