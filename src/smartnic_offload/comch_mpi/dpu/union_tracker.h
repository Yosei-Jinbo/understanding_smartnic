#ifndef UNION_TRACKER_H
#define UNION_TRACKER_H

#include <stdbool.h>

/* DPU Union Tracker (union_tracker.c)
 *   interval union で AG/RS/overlap の busy 時間を wall clock で測る */

void dpu_union_tracker_enter(bool is_ag);
void dpu_union_tracker_exit(bool is_ag);
void dpu_union_tracker_print(int rank);

#endif /* UNION_TRACKER_H */
