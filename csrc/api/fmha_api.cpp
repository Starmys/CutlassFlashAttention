// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


#include "torch/extension.h"


void fmha_forward_inference(
  at::Tensor Q, // B, Nt, H, D
  at::Tensor K, // B, Ns, H, D
  at::Tensor V, // B, Ns, H, D
  at::Tensor O, // B, Nt, H, D
  float scale,
  bool causal
);


void fmha_forward_training(
  at::Tensor Q, // B, Nt, H, D
  at::Tensor K, // B, Ns, H, D
  at::Tensor V, // B, Ns, H, D
  at::Tensor O, // B, Nt, H, D
  at::Tensor lse, // B, H, Nt
  float scale,
  bool causal
);


void fmha_backward(
  at::Tensor Q, // B, Nt, H, D
  at::Tensor K, // B, Ns, H, D
  at::Tensor V, // B, Ns, H, D
  at::Tensor O, // B, Nt, H, D
  at::Tensor dQ, // B, Nt, H, D
  at::Tensor dK, // B, Ns, H, D
  at::Tensor dV, // B, Ns, H, D
  at::Tensor dO, // B, Nt, H, D
  at::Tensor lse, // B, H, Nt
  at::Tensor delta, // B, H, Nt
  float scale,
  bool causal
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_inference", &fmha_forward_inference, "FHMA forward function for inference");
  m.def("forward_training", &fmha_forward_training, "FHMA forward function for training");
  m.def("backward", &fmha_backward, "FHMA backward function");
}
