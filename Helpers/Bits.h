/**
 * @file Bits.h
 * @author GreenBean
 * @brief Bit manipulation helper macros.
 * @version 0.1
 * @date 2026-10-17
 */
#pragma once
#define BT_BITS_OP_SET_BIT(x, n) ((x) |= (1u << (n)))
#define BT_BITS_OP_CLEAR_BIT(x, n) ((x) &= ~(1u << (n)))
#define BT_BITS_OP_CHECK_BIT(x, n) (((x) >> (n)) & 1u)