import type { UsageTotals } from '../api/types';

export const fmtCost = (value = 0) => {
  if (value !== 0 && Math.abs(value) < 0.00000001) return value < 0 ? '−<¥0.00000001' : '<¥0.00000001';
  return '¥' + value.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 8 });
};
export const pendingCalls = (item: UsageTotals) =>
  (item.unpriced_calls || 0) + (item.unverified_zero_calls || 0);
export const aggregateAmount = (item: UsageTotals) =>
  pendingCalls(item) > 0 && pendingCalls(item) === item.calls ? '—' : fmtCost(item.cost_cny);
