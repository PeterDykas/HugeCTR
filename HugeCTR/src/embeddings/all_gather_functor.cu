/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"

namespace HugeCTR {
/**
 * collection communication: all_gather.
 * @param send_count the count of elements will be sent.
 * @param send_tensors the send tensors of multi GPUs.
 * @param recv_tensors the recv tensors of multi GPUs.
 * @param device_resources all gpus device resources.
 * @param context gpu device context, for switching device.
 */
template <typename Type>
void SparseEmbeddingFunctors::all_gather(size_t send_count, const TensorPtrs<Type> &send_tensors,
                                         const TensorPtrs<Type> &recv_tensors,
                                         const GPUResourceGroup &device_resources) {
  size_t local_gpu_count = device_resources.size();
  size_t total_gpu_count = device_resources.get_total_gpu_count();

  // need to know the Type
  ncclDataType_t type;
  switch (sizeof(Type)) {
    case 2:
      type = ncclHalf;
      break;
    case 4:
      type = ncclFloat;
      break;
    default:
      CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
  }

  // for multi GPUs, use NCCL to do All-Gather
  if (total_gpu_count > 1) {
    CK_NCCL_THROW_(ncclGroupStart());
    for (size_t id = 0; id < local_gpu_count; id++) {
      CK_NCCL_THROW_(ncclAllGather(send_tensors[id]->get_ptr(),  // send buff
                                   recv_tensors[id]->get_ptr(),  // recv buff
                                   send_count, type, device_resources[id].get_nccl(),
                                   device_resources[id].get_stream()));
    }
    CK_NCCL_THROW_(ncclGroupEnd());
  }
  // for single GPU, just do memcpyD2D
  else {  // total_gpu_count == 1
    CudaDeviceContext context(device_resources[0].get_device_id());
    CK_CUDA_THROW_(cudaMemcpyAsync(recv_tensors[0]->get_ptr(), send_tensors[0]->get_ptr(),
                                   send_count * sizeof(Type), cudaMemcpyDeviceToDevice,
                                   device_resources[0].get_stream()));
  }

  return;
}

template void SparseEmbeddingFunctors::all_gather<float>(size_t send_count,
                                                         const TensorPtrs<float> &send_tensors,
                                                         const TensorPtrs<float> &recv_tensors,
                                                         const GPUResourceGroup &device_resources);

template void SparseEmbeddingFunctors::all_gather<__half>(size_t send_count,
                                                          const TensorPtrs<__half> &send_tensors,
                                                          const TensorPtrs<__half> &recv_tensors,
                                                          const GPUResourceGroup &device_resources);

}  // namespace HugeCTR