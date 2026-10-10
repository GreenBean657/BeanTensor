#pragma once
#define SET_BIT(x, n) ((x) |= (1u << (n)))
#define CLEAR_BIT(x, n) ((x) &= ~(1u << (n)))
#define CHECK_BIT(x, n) (((x) >> (n)) & 1u)