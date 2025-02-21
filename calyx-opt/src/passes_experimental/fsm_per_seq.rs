use crate::traversal::{Action, Named, VisResult, Visitor};
use calyx_ir::{self as ir, LibrarySignatures};

#[derive(Default)]
/// Transforms each sequential block into a

pub struct FsmPerSeq {}

impl Named for FsmPerSeq {
    fn name() -> &'static str {
        "fsm-per-seq"
    }

    fn description() -> &'static str {
        "Add a @new_fsm to each sequential block"
    }
}

impl Visitor for FsmPerSeq {
    /// Add a new FSM to each sequential block
    fn start_seq(
        &mut self,
        s: &mut calyx_ir::Seq,
        _comp: &mut calyx_ir::Component,
        _sigs: &LibrarySignatures,
        _comps: &[calyx_ir::Component],
    ) -> VisResult {
        for con in s.stmts.iter() {
            match con {
                ir::Control::Seq(_data) => {
                    return Ok(Action::Continue);
                }

                ir::Control::Par(_data) => {
                    return Ok(Action::Continue);
                }

                ir::Control::If(_data) => {
                    return Ok(Action::Continue);
                }

                ir::Control::While(_data) => {
                    return Ok(Action::Continue);
                }

                _ => {}
            }
        }

        s.attributes.insert(ir::BoolAttr::NewFSM, 1);
        Ok(Action::Continue)
    }
}
