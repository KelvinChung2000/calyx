use crate::exec::State;

use super::{
    super::{OpRef, Operation, StateRef},
    FindPlan, Step,
};
use cranelift_entity::PrimaryMap;

#[derive(Debug, Default)]
pub struct EnumeratePlanner {}
impl EnumeratePlanner {
    /// The max number of ops in a searched for plan.
    const MAX_PLAN_LEN: u32 = 7;

    pub fn new() -> Self {
        EnumeratePlanner {}
    }

    /// Returns `true` if executing `plan` will take `start` to `end`, going through all ops in `through`.
    ///
    /// This function assumes all required inputs to `plan` exist or will be generated by `plan`.
    fn valid_plan(plan: &[Step], end: &[StateRef], through: &[OpRef]) -> bool {
        // Check all states in `end` are created.
        let end_created = end
            .iter()
            .all(|s| plan.iter().any(|(_, states)| states.contains(s)));

        // FIXME: Currently this checks that an outputs of an op specified by though is used.
        // However, it's possible that the only use of this output by another op whose outputs
        // are all unused. This means the plan doesn't actually use the specified op. but this
        // code reports it would.
        let through_used = through.iter().all(|t| {
            plan.iter()
                .any(|(op, used_states)| op == t && !used_states.is_empty())
        });

        end_created && through_used
    }

    /// A recursive function to generate all sequences prefixed by `plan` and containing `len` more
    /// `Steps`. Returns a sequence such that applying `valid_plan` to the sequence results in `true.
    /// If no such sequence exists, then `None` is returned.
    ///
    /// `start` is the base inputs which can be used for ops.
    /// `end` is the states to be generated by the return sequence of ops.
    /// `ops` contains all usable operations to construct `Step`s from.
    fn try_paths_of_length(
        plan: &mut Vec<Step>,
        len: u32,
        start: &[StateRef],
        end: &[StateRef],
        through: &[OpRef],
        ops: &PrimaryMap<OpRef, Operation>,
    ) -> Option<Vec<Step>> {
        // Base case of the recursion. As `len == 0`, the algorithm reduces to applying `good` to
        // `plan.
        if len == 0 {
            return if Self::valid_plan(plan, end, through) {
                Some(plan.clone())
            } else {
                None
            };
        }

        // Try adding every op to the back of the current `plan`. Then recurse on the subproblem.
        for op_ref in ops.keys() {
            // Check previous ops in the plan to see if any generated an input to `op_ref`.
            let all_generated = ops[op_ref].input.iter().all(|input| {
                // Check the outputs of ops earlier in the plan can be used as inputs to `op_ref`.
                // `plan` is reversed so the latest versions of states are used.
                plan.iter_mut().rev().any(|(o, _used_outputs)|
                    ops[*o].output.contains(input)
                )
                // As well as being generated in `plan`, `input` could be given in `start`.
                || start.contains(input)
            });

            // If this op cannot be uesd in the `plan` try a different one.
            if !all_generated {
                continue;
            }

            // Mark all used outputs.
            let used_outputs_idxs: Vec<_> = ops[op_ref]
                .input
                .iter()
                .filter_map(|input| {
                    // Get indicies of `Step`s whose `used_outputs` must be modified.
                    plan.iter()
                        .rev()
                        .position(|(o, used_outputs)| {
                            // `op_ref`'s op now uses the input of the previous op in the plan.
                            // This should be noted in `used_outputs`.
                            !used_outputs.contains(input)
                                && ops[*o].output.contains(input)
                        })
                        .map(|i| (input, plan.len() - i - 1))
                })
                .collect();

            for &(&input, i) in &used_outputs_idxs {
                plan[i].1.push(input);
            }

            // Mark all outputs in `end` as used because they are used (or at least requested) by
            // `end`.
            let outputs = ops[op_ref].output.clone().into_iter();
            let used_outputs =
                outputs.filter(|s| end.contains(s)).collect::<Vec<_>>();

            // Recurse! Now that `len` has been reduced by one, see if this new problem has a
            // solution.
            plan.push((op_ref, used_outputs));
            if let Some(plan) = Self::try_paths_of_length(
                plan,
                len - 1,
                start,
                end,
                through,
                ops,
            ) {
                return Some(plan);
            }

            // The investigated plan didn't work.
            // Pop off the attempted element.
            plan.pop();

            // Revert modifications to `used_outputs`.
            for &(_, i) in &used_outputs_idxs {
                plan[i].1.pop();
            }
        }

        // No sequence of `Step`s found :(.
        None
    }

    /// Returns a sequence of `Step`s to transform `start` to `end`. The `Step`s are guaranteed to
    /// contain all ops in `through`. If no such sequence exists, `None` is returned.
    ///
    /// `ops` is a complete list of operations.
    fn find_plan(
        start: &[StateRef],
        end: &[StateRef],
        through: &[OpRef],
        ops: &PrimaryMap<OpRef, Operation>,
    ) -> Option<Vec<Step>> {
        // Try all sequences of ops up to `MAX_PATH_LEN`. At that point, the computation starts to
        // become really big.
        for len in 1..=Self::MAX_PLAN_LEN {
            if let Some(plan) = Self::try_paths_of_length(
                &mut vec![],
                len,
                start,
                end,
                through,
                ops,
            ) {
                return Some(plan);
            }
        }

        // No sequence of `Step`s found :(.
        None
    }
}

impl FindPlan for EnumeratePlanner {
    fn find_plan(
        &self,
        start: &[StateRef],
        end: &[StateRef],
        through: &[OpRef],
        ops: &PrimaryMap<OpRef, Operation>,
        _states: &PrimaryMap<StateRef, State>,
    ) -> Option<Vec<Step>> {
        Self::find_plan(start, end, through, ops)
    }
}