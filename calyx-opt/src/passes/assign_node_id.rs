use crate::passes;
use crate::traversal::{
    Action, ConstructVisitor, Named, VisResult, Visitor,
};
use calyx_ir::{
    self as ir, Assignment, BoolAttr, GetAttributes, LibrarySignatures, NumAttr, Printer, StaticTiming, RRC
};

use calyx_utils::CalyxResult;

const NODE_ID: ir::Attribute =
    ir::Attribute::Internal(ir::InternalAttr::NODE_ID);

pub struct AssignNodeId;

impl ConstructVisitor for AssignNodeId {
    fn from(_ctx: &ir::Context) -> CalyxResult<Self>
    {
        Ok(AssignNodeId)
    }

    fn clear_data(&mut self) {
        /* All data can be transferred between components */
    }
}

impl Named for AssignNodeId {
    fn name() -> &'static str {
        "assign-node-id"
    }

    fn description() -> &'static str {
        "Assigns unique node IDs to components in the design"
    }
}

/// Adds the @NODE_ID attribute to [ir::Enable] and [ir::Par].
/// Each [ir::Enable] gets a unique label within the context of a child of
/// a [ir::Par] node.
/// Furthermore, if an if/while/seq statement is labeled with a `new_fsm` attribute,
/// then it will get its own unique label. Within that if/while/seq, each enable
/// will get its own unique label within the context of that if/while/seq (see
/// example for clarification).
///
/// ## Example:
/// ```
/// seq { A; B; par { C; D; }; E; @new_fsm seq {F; G; H}}
/// ```
/// gets the labels:
/// ```
/// seq {
///   @NODE_ID(1) A; @NODE_ID(2) B;
///   @NODE_ID(3) par {
///     @NODE_ID(0) C;
///     @NODE_ID(0) D;
///   }
///   @NODE_ID(4) E;
///   @NODE_ID(5) seq{
///     @NODE_ID(0) F;
///     @NODE_ID(1) G;
///     @NODE_ID(2) H;
///   }
/// }
/// ```
///
/// These identifiers are used by the compilation methods [calculate_states_recur]
/// and [control_exits].
/// These identifiers are used by the compilation methods [calculate_states_recur]
/// and [control_exits].
fn compute_unique_state_ids(con: &mut ir::Control, cur_state: u64) -> u64 {
    match con {
        ir::Control::Enable(ir::Enable { attributes, .. }) => {
            attributes.insert(NODE_ID, cur_state);
            cur_state + 1
        }
        ir::Control::Par(ir::Par { stmts, attributes }) => {
            attributes.insert(NODE_ID, cur_state);
            stmts.iter_mut().for_each(|stmt| {
                compute_unique_state_ids(stmt, 0);
            });
            cur_state + 1
        }
        ir::Control::Seq(ir::Seq { stmts, attributes }) => {
            let new_fsm = attributes.has(ir::BoolAttr::NewFSM);
            // if new_fsm is true, then insert attribute at the seq, and then
            // start over counting states from 0
            let mut cur = if new_fsm{
                attributes.insert(NODE_ID, cur_state);
                0
            } else {
                cur_state
            };
            stmts.iter_mut().for_each(|stmt| {
                cur = compute_unique_state_ids(stmt, cur);
            });
            // If new_fsm is true then we want to return cur_state + 1, since this
            // seq should really only take up 1 "state" on the "outer" fsm
            if new_fsm{
                cur_state + 1
            } else {
                cur
            }
        }
        ir::Control::If(ir::If {
            tbranch, fbranch, attributes, ..
        }) => {
            let new_fsm = attributes.has(ir::BoolAttr::NewFSM);
            // if new_fsm is true, then we want to add an attribute to this
            // control statement
            if new_fsm {
                attributes.insert(NODE_ID, cur_state);
            }
            // If the program starts with a branch then branches can't get
            // the initial state.
            // Also, if new_fsm is true, we want to start with state 1 as well:
            // we can't start at 0 for the reason mentioned above
            let cur = if new_fsm || cur_state == 0 {
                1
            } else {
                cur_state
            };
            let tru_nxt = compute_unique_state_ids(
                tbranch, cur
            );
            let false_nxt = compute_unique_state_ids(
                fbranch, tru_nxt
            );
            // If new_fsm is true then we want to return cur_state + 1, since this
            // if stmt should really only take up 1 "state" on the "outer" fsm
            if new_fsm {
                cur_state + 1
            } else {
                false_nxt
            }
        }
        ir::Control::While(ir::While { body, attributes, .. }) => {
            let new_fsm = attributes.has(ir::BoolAttr::NewFSM);
            // if new_fsm is true, then we want to add an attribute to this
            // control statement
            if new_fsm{
                attributes.insert(NODE_ID, cur_state);
            }
            // If the program starts with a branch then branches can't get
            // the initial state.
            // Also, if new_fsm is true, we want to start with state 1 as well:
            // we can't start at 0 for the reason mentioned above
            let cur = if new_fsm || cur_state == 0 {
                1
            } else {
                cur_state
            };
            let body_nxt = compute_unique_state_ids(body, cur);
            // If new_fsm is true then we want to return cur_state + 1, since this
            // while loop should really only take up 1 "state" on the "outer" fsm
            if new_fsm{
                cur_state + 1
            } else {
                body_nxt
            }
        }
        ir::Control::FSMEnable(_) => unreachable!("shouldn't encounter fsm node"),
        ir::Control::Empty(_) => cur_state,
        ir::Control::Repeat(_) => unreachable!("`repeat` statements should have been compiled away. Run `{}` before this pass.", passes::CompileRepeat::name()),
        ir::Control::Invoke(_) => unreachable!("`invoke` statements should have been compiled away. Run `{}` before this pass.", passes::CompileInvoke::name()),
        ir::Control::Static(sc) => {
            if let ir::StaticControl::Enable(ir::StaticEnable{attributes,..}) = sc {
                if attributes.has(NODE_ID) {
                    attributes.remove(NODE_ID);
                }
                attributes.insert(NODE_ID, cur_state);

                // if with new fsm then we create 1 state, otherwise we keep the latency amount of state
                if attributes.has(ir::BoolAttr::NewFSM) {
                    cur_state + 1
                } else {
                    cur_state + sc.get_latency()
                }
            } else {
                unreachable!("static control should have been compiled away. Run the static-inline passes before this pass")
            }
        }
    }
}


impl Visitor for AssignNodeId {
    fn start(
        &mut self,
        comp: &mut calyx_ir::Component,
        _sigs: &LibrarySignatures,
        _comps: &[calyx_ir::Component],
    ) -> VisResult {
        let mut con = comp.control.borrow_mut();
        compute_unique_state_ids(&mut con, 0);
        Ok(Action::Continue)
    }
}