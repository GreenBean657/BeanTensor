/**
 * @file Autograd.h
 * @author GreenBean
 * @brief Entry point for the autograd backward ops (cpu vs gpu): gemm_g,
 * bias_add_g, relu_g, softmax_ce_g, sum_g, mean_g, the elementwise add/sub/
 * mul/div/pow (tensor-tensor and tensor-scalar) and exp/log backward nodes,
 * plus the shared gradient plumbing (grad accumulation, last-two-dims
 * transpose for GEMM backward, broadcast gradient reduction for elementwise
 * backward). The *_g_node functions match Node::_backward_fn's signature and
 * are what the forward ops store when they wire the computation graph.
 * @version 0.3
 * @date 2026-10-25
 */
#pragma once

#include "Math/Autograd/GradAcc/GradAcc.h"
#include "Math/Autograd/ReluG/ReluG.h"
#include "Math/Autograd/BiasAddG/BiasAddG.h"
#include "Math/Autograd/SoftmaxCEG/SoftmaxCEG.h"
#include "Math/Autograd/TransposeLast2/TransposeLast2.h"
#include "Math/Autograd/GemmG/GemmG.h"
#include "Math/Autograd/SumG/SumG.h"
#include "Math/Autograd/MeanG/MeanG.h"
#include "Math/Autograd/ElementwiseG/ElementwiseG.h"
