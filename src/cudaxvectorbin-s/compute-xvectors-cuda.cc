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
#include "cudafeat/feature-spectral-cuda.h"
#include "feat/wave-reader.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-allocator.h"


bool ComputeMFCC(const WaveData &wave_data, BaseFloat min_duration, int32 channel,
		CudaSpectralFeatures& mfcc, Matrix<BaseFloat>& features) {

	if (wave_data.Duration() < min_duration) {
	        KALDI_WARN << "File: " << utt << " is too short ("
	                   << wave_data.Duration() << " sec): producing no output.";
	        return false;
	      }
	      int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
	      {  // This block works out the channel (0=left, 1=right...)
	        KALDI_ASSERT(num_chan > 0);  // should have been caught in
	        // reading code if no channels.
	        if (channel == -1) {
	          this_chan = 0;
	          if (num_chan != 1)
	            KALDI_WARN << "Channel not specified but you have data with "
	                       << num_chan  << " channels; defaulting to zero";
	        } else {
	          if (this_chan >= num_chan) {
	            KALDI_WARN << "File with id " << utt << " has "
	                       << num_chan << " channels but you specified channel "
	                       << channel << ", producing no output.";
	            return false;
	          }
	        }
	      }

	      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);


	      CuVector<BaseFloat> cu_waveform(waveform);
	      CuMatrix<BaseFloat> cu_features;
	      mfcc.ComputeFeatures(cu_waveform, wave_data.SampFreq(), 1.0, &cu_features);
	      features.Resize(cu_features.NumRows(), cu_features.NumCols());
	      features.CopyFromMat(cu_features);

	return true;

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Create xvectors files.\n"
        "Usage:  compute-xvectors-cuda [options...] <wav-rspecifier> <xvectors-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);
    MfccOptions mfcc_opts;
    bool subtract_mean = false;
    int32 channel = -1;
    BaseFloat min_duration = 0.0;
    // Define defaults for gobal options
    std::string output_format = "kaldi";

    // Register the MFCC option struct
    mfcc_opts.Register(&po);

    // Register the options
    po.Register("output-format", &output_format, "Format of the output "
                "files [kaldi, htk]");
    po.Register("subtract-mean", &subtract_mean, "Subtract mean of each "
                "feature file [CMS]; not recommended to do it this way. ");

    po.Register("utt2spk", &utt2spk_rspecifier, "Utterance to speaker-id map "
                "rspecifier (if doing VTLN and you have warps per speaker)");
    po.Register("channel", &channel, "Channel to extract (-1 -> expect mono, "
                "0 -> left, 1 -> right)");
    po.Register("min-duration", &min_duration, "Minimum duration of segments "
                "to process (in seconds).");
    RegisterCuAllocatorOptions(&po);

    CuDevice::RegisterDeviceOptions(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    g_cuda_allocator.SetOptions(g_allocator_options);
    CuDevice::Instantiate().SelectGpuId("yes");
    CuDevice::Instantiate().AllowMultithreading();


    std::string wav_rspecifier = po.GetArg(1);

    std::string output_wspecifier = po.GetArg(2);

    CudaSpectralFeatures mfcc(mfcc_opts);

    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.
    TableWriter<HtkMatrixHolder> htk_writer;



    int32 num_utts = 0, num_success = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;

      Matrix<BaseFloat> features;

      std::string utt = reader.Key();
      const WaveData &wave_data = reader.Value();


      if (ComputeMFCC(&wave_data, min_duration, channel, &mfcc, &features) == false){
    	  continue;
      }

      if (num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
      num_success++;
    }

    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

