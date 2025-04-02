use crate::passes;
use crate::traversal::{
    Action, ConstructVisitor, Named, ParseVal, PassOpt, VisResult, Visitor,
};
use calyx_ir::{
    self as ir, Assignment, BoolAttr, Canonical, Cell, GetAttributes, LibrarySignatures, NumAttr, Printer, StaticTiming, RRC
};

use calyx_ir::{build_assignments, guard, structure, Id};
use calyx_utils::Error;
use calyx_utils::{CalyxResult, OutputFile};
use ir::Nothing;
use itertools::Itertools;
use petgraph::{dot, Graph};
use petgraph::graph::DiGraph;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{repeat, Write};
use std::rc::Rc;

const NODE_ID: ir::Attribute =
    ir::Attribute::Internal(ir::InternalAttr::NODE_ID);


/// Computes the exit edges of a given [ir::Control] program.
///
/// ## Example
/// In the following Calyx program:
/// ```
/// while comb_reg.out {
///   seq {
///     @NODE_ID(4) incr;
///     @NODE_ID(5) cond0;
///   }
/// }
/// ```
/// The exit edge is is `[(5, cond0[done])]` indicating that the state 5 exits when the guard
/// `cond0[done]` is true.
///
/// Multiple exit points are created when conditions are used:
/// ```
/// while comb_reg.out {
///   @NODE_ID(7) incr;
///   if comb_reg2.out {
///     @NODE_ID(8) tru;
///   } else {
///     @NODE_ID(9) fal;
///   }
/// }
/// ```
/// The exit set is `[(8, tru[done] & !comb_reg.out), (9, fal & !comb_reg.out)]`.
fn control_exits(con: &ir::Control, exits: &mut Vec<PredEdge>) {
    match con {
        ir::Control::Empty(_) => {},
        ir::Control::Enable(ir::Enable { group, attributes }) => {
            let cur_state = attributes.get(NODE_ID).unwrap();
            // exits.push((cur_state, guard!(group["done"])))
            exits.push((cur_state, *group.borrow().done_cond().guard.clone()));
        },
        ir::Control::FSMEnable(ir::FSMEnable{attributes, fsm}) => {
            let cur_state = attributes.get(NODE_ID).unwrap();
            exits.push((cur_state, guard!(fsm["done"])))
        },
        ir::Control::Seq(ir::Seq { stmts, .. }) => {
            if let Some(stmt) = stmts.last() { control_exits(stmt, exits) }
        }
        ir::Control::If(ir::If {
            tbranch, fbranch, ..
        }) => {
            control_exits(
                tbranch, exits,
            );
            control_exits(
                fbranch, exits,
            )
        }
        ir::Control::While(ir::While { body, port, .. }) => {
            let mut loop_exits = vec![];
            control_exits(body, &mut loop_exits);
            // Loop exits only happen when the loop guard is false
            exits.extend(loop_exits.into_iter().map(|(s, g)| {
                (s, g & !ir::Guard::from(port.clone()))
            }));
        },
        ir::Control::Repeat(_) => unreachable!("`repeat` statements should have been compiled away. Run `{}` before this pass.", passes::CompileRepeat::name()),
        ir::Control::Invoke(_) => unreachable!("`invoke` statements should have been compiled away. Run `{}` before this pass.", passes::CompileInvoke::name()),
        ir::Control::Par(_) => unreachable!(),
        ir::Control::Static(sc) => {
            if let ir::StaticControl::Enable(ir::StaticEnable{attributes, ..}) = sc {
                let cur_state = attributes.get(NODE_ID).unwrap();
                exits.push((cur_state + sc.get_latency(), ir::Guard::True));
            } else {
                unreachable!("static control should have been compiled away. Run the static-inline passes before this pass")
            }
        }
    }
}

/// Represents the dyanmic execution schedule of a control program.
struct Schedule<'b, 'a: 'b> {
    /// A mapping from groups to corresponding FSM state ids
    pub groups_to_states: HashSet<FSMStateInfo>,
    /// Assigments that should be enabled in a given state.
    pub enables: BTreeMap<u64, Vec<ir::Assignment<Nothing>>>,
    /// FSMs that should be triggered in a given state.
    pub fsm_enables: BTreeMap<u64, Vec<ir::Assignment<Nothing>>>,
    /// Transition from one state to another when the guard is true.
    pub transitions: Vec<(u64, u64, ir::Guard<Nothing>)>,
    /// The component builder. The reference has a shorter lifetime than the builder itself
    /// to allow multiple schedules to use the same builder.
    pub builder: &'b mut ir::Builder<'a>,
}

/// Information to serialize for profiling purposes
#[derive(PartialEq, Eq, Hash, Clone, Serialize)]
enum ProfilingInfo {
    SingleEnable(SingleEnableInfo),
}

/// Information to be serialized for a group that isn't managed by a FSM
/// This can happen if the group is the only group in a control block or a par arm
#[derive(PartialEq, Eq, Hash, Clone, Serialize)]
struct SingleEnableInfo {
    #[serde(serialize_with = "id_serialize_passthrough")]
    pub component: Id,
    #[serde(serialize_with = "id_serialize_passthrough")]
    pub group: Id,
}

/// Mapping of FSM state ids to corresponding group names
#[derive(PartialEq, Eq, Hash, Clone, Serialize)]
struct FSMStateInfo {
    id: u64,
    #[serde(serialize_with = "id_serialize_passthrough")]
    group: Id,
}

fn id_serialize_passthrough<S>(id: &Id, ser: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    id.to_string().serialize(ser)
}

impl<'b, 'a> From<&'b mut ir::Builder<'a>> for Schedule<'b, 'a> {
    fn from(builder: &'b mut ir::Builder<'a>) -> Self {
        Schedule {
            groups_to_states: HashSet::new(),
            enables: BTreeMap::new(),
            fsm_enables: BTreeMap::new(),
            transitions: Vec::new(),
            builder,
        }
    }
}

impl<'b, 'a> Schedule<'b, 'a> {
    /// Validate that all states are reachable in the transition graph.
    fn validate(&self) {
        let graph: petgraph::Graph<(), u32> = DiGraph::<(), u32>::from_edges(
            self.transitions
                .iter()
                .map(|(s, e, _)| (*s as u32, *e as u32)),
        );

        debug_assert!(
            petgraph::algo::connected_components(&graph) == 1,
            "State transition graph for {} has unreachable states (graph has more than one connected component)\n{:?}.", 
            self.builder.component.name,
            petgraph::dot::Dot::with_config(&graph, &[petgraph::dot::Config::EdgeNoLabel, petgraph::dot::Config::NodeNoLabel])
        );
    }

    /// Return the max state in the transition graph
    fn last_state(&self) -> u64 {
        self.transitions
            .iter()
            .max_by_key(|(_, s, _)| s)
            .expect("Schedule::transition is empty!")
            .1
    }

    /// Print out the current schedule
    fn display(&self, group: String) {
        let out = &mut std::io::stdout();
        writeln!(out, "======== {} =========", group).unwrap();
        self.enables
            .iter()
            .sorted_by(|(k1, _), (k2, _)| k1.cmp(k2))
            .for_each(|(state, assigns)| {
                writeln!(out, "{}:", state).unwrap();
                assigns.iter().for_each(|assign| {
                    Printer::write_assignment(assign, 2, out).unwrap();
                    writeln!(out).unwrap();
                })
            });
        writeln!(out, "{}:\n  <end>", self.last_state()).unwrap();
        writeln!(out, "transitions:").unwrap();
        self.transitions
            .iter()
            .sorted_by(|(k1, _, _), (k2, _, _)| k1.cmp(k2))
            .for_each(|(i, f, g)| {
                writeln!(out, "  ({}, {}): {}", i, f, Printer::guard_str(g))
                    .unwrap();
            });
    }

    fn realize_fsm(self, dump_fsm: bool, dump_dot: &String) -> RRC<ir::FSM> {
        // ensure schedule is valid
        self.validate();

        let mut tran_clone = vec![];
        for t in self.transitions.iter() {
            tran_clone.push((t.0, t.1));
        }
        // compute final state and fsm_size, and register initial fsm
        let fsm: Rc<std::cell::RefCell<calyx_ir::FSM>> = self.builder.add_fsm("fsm");

        if dump_fsm {
            println!("==== {} ====", fsm.borrow().name());
            self.display(format!(
                "{}:{}",
                self.builder.component.name,
                fsm.borrow().name()
            ));
        }

        
        if !dump_dot.is_empty() {
            let graph: Graph<(), u32> = DiGraph::<(), u32>::from_edges(
                self.transitions
                .iter()
                .map(|(s, e, _)| (*s as u32, *e as u32)),
            );
            let mut dot_file =
            File::create(dump_dot).expect("Unable to create file");
            writeln!(
                dot_file,
                "{:?}",
                dot::Dot::with_config(
                    &graph,
                    &[dot::Config::EdgeNoLabel, dot::Config::NodeNoLabel]
                )
            )
            .expect("Unable to write to dot file");
        }
    
        // map each source state to a list of conditional transitions
        let mut transitions_map: BTreeMap<u64, Vec<(ir::Guard<Nothing>, u64)>> =
            BTreeMap::new();
        self.transitions.into_iter().for_each(
            |(s, e, g)| match transitions_map.get_mut(&(s + 1)) {
                Some(next_states) => next_states.push((g.clone(), e + 1)),
                None => {
                    transitions_map.insert(s + 1, vec![(g.clone(), e + 1)]);
                }
            },
        );

        // push the cases of the fsm to the fsm instantiation
        let (mut transitions, mut assignments): (
            VecDeque<ir::Transition>,
            VecDeque<Vec<ir::Assignment<Nothing>>>,
        ) = transitions_map
            .into_iter()
            .sorted_by(|(s1, _), (s2, _)| s1.cmp(s2))
            .map(|(state, mut cond_dsts)| {
                let assigns: Vec<calyx_ir::Assignment<Nothing>> =
                    match self.fsm_enables.get(&(state - 1)) {
                        None => {
                            vec![]
                        },
                        Some(assigns) => {
                            assigns.clone()
                        }
                    };
                if cond_dsts.len() == 1 && cond_dsts[0].0.is_true() {
                    // if only have one true guard, then it is an unconditional transition
                    (ir::Transition::Unconditional(state + 1), assigns)
                } else {
                    // Add a self-loop transition that activates when all other guards are not met;
                    // This must be at the end of the conditional destinations vector to serve as the default case.
                    cond_dsts.push((ir::Guard::True, state));
                    (ir::Transition::Conditional(cond_dsts), assigns)
                }
            })
            .unzip();
        // insert transition condition from 0 to 1
        let true_guard = ir::Guard::True;
        assignments.push_front(vec![]);
        transitions.push_front(ir::Transition::Conditional(vec![
            (guard!(fsm["start"]), 1),
            (true_guard.clone(), 0),
        ]));

        // insert transition from final calc state to `done` state
        let signal_on = self.builder.add_constant(1, 1);
        let assign = build_assignments!(self.builder;
            fsm["done"] = true_guard ? signal_on["out"];
        );
        assignments.push_back(assign.to_vec());
        transitions.push_back(ir::Transition::Unconditional(0));
        fsm.borrow_mut().assignments.extend(assignments);
        fsm.borrow_mut().transitions.extend(transitions);

        // register group enables dependent on fsm state as assignments in the
        // relevant state's assignment section
        // println!("enables:");
        // self.enables.iter()
        //     .sorted_by(|(k1, _), (k2, _)| k1.cmp(k2))
        //     .for_each(|(state, assigns)| {
        //         println!("  State {}: ", state);
        //         assigns.iter().for_each(|assign| {
        //             print!("    ");
        //             let out = &mut std::io::stdout();
        //             Printer::write_assignment(assign, 4, out).unwrap();
        //             println!();
        //         });
        //     });
        let mut repeat_set: BTreeMap<Canonical, Rc<RefCell<Cell>>> = BTreeMap::new();
        self.enables.into_iter().for_each(|(state, state_enables)| {
            let mut result = vec![];
            for assign in state_enables.iter() {
                let fsm_out_wire = if repeat_set.contains_key(&assign.dst.borrow().canonical()) {
                    Rc::clone(repeat_set.get(&assign.dst.borrow().canonical()).unwrap())
                } else {
                    let name = assign.dst.borrow().canonical().to_string();
                    let out_wire = self.builder.add_primitive(
                        format!("{}_{}", fsm.borrow().name().to_string(), name.replace(".", "_")), 
                                                "std_wire", 
                                                &[assign.dst.borrow().width.clone()]);
                    repeat_set.insert(assign.dst.borrow().canonical(), Rc::clone(&out_wire));
                    out_wire
                };
                result.push(ir::Assignment {
                    src: assign.src.clone(),
                    dst: fsm_out_wire.borrow().get("in"),
                    attributes: assign.attributes.clone(),
                    guard: Box::new(ir::Guard::True),
                });
                fsm_out_wire.borrow_mut().attributes = assign.dst.borrow().attributes.clone();
                fsm_out_wire.borrow_mut().get("out").borrow_mut().attributes = assign.dst.borrow().attributes.clone();
                fsm_out_wire.borrow_mut().get("in").borrow_mut().attributes = assign.dst.borrow().attributes.clone();
                self.builder.add_continuous_assignments(vec![
                    ir::Assignment {
                        src: fsm_out_wire.borrow().get("out"),
                        dst: assign.dst.clone(),
                        attributes: assign.attributes.clone(),
                        // guard: Box::new(ir::Guard::And(assign.guard.clone(), 
                        //                                Box::new(guard!(fsm["start"]))).simplify()
                        guard: assign.guard.clone(),
                    }
                ]);
            }
            fsm.borrow_mut()
                .extend_state_assignments(state + 1, result);
        });
        fsm
    }
}

/// Represents an edge from a predeccesor to the current control node.
/// The `u64` represents the FSM state of the predeccesor and the guard needs
/// to be true for the predeccesor to transition to the current state.
type PredEdge = (u64, ir::Guard<Nothing>);

impl Schedule<'_, '_> {
    /// Recursively build an dynamic finite state machine represented by a [Schedule].
    /// Does the following, given an [ir::Control]:
    ///     1. If needed, add transitions from predeccesors to the current state.
    ///     2. Enable the groups in the current state
    ///     3. Calculate [PredEdge] implied by this state
    ///     4. Return [PredEdge] and the next state.
    /// Another note: the functions calc_seq_recur, calc_while_recur, and calc_if_recur
    /// are functions that `calculate_states_recur` uses for when con is a seq, while,
    /// and if respectively. The reason why they are defined as separate functions is because we
    /// need to call `calculate_seq_recur` (for example) directly when we are in `finish_seq`
    /// since `finish_seq` only gives us access to a `& mut seq` type, not a `& Control`
    /// type.
    fn calculate_states_recur(
        // Current schedule.
        &mut self,
        con: &ir::Control,
        // The set of previous states that want to transition into cur_state
        preds: Vec<PredEdge>,
        // True if early_transitions are allowed
        early_transitions: bool,
        // True if the `@fast` attribute has successfully been applied to the parent of this control
        has_fast_guarantee: bool,
    ) -> CalyxResult<Vec<PredEdge>> {
        match con {
        ir::Control::FSMEnable(ir::FSMEnable {fsm, attributes}) => {
            let cur_state = attributes.get(NODE_ID).unwrap_or_else(|| panic!("Group `{}` does not have state_id information", fsm.borrow().name()));
            let (cur_state, prev_states) = if preds.len() == 1 && preds[0].1.is_true() && cur_state == 0 {
                (preds[0].0, vec![])
            } else {
                (cur_state, preds)
            };

            // Add group to mapping for emitting group JSON info
            self.groups_to_states.insert(FSMStateInfo { id: cur_state, group: fsm.borrow().name() });

            let not_done = ir::Guard::True;
            let signal_on = self.builder.add_constant(1, 1);

            // Activate this fsm in the current state
            let en_go : [ir::Assignment<Nothing>; 1] = build_assignments!(self.builder;
                fsm["start"] = not_done ? signal_on["out"];
            );

            self.fsm_enables.entry(cur_state).or_default().extend(en_go);

            // Enable FSM to be triggered by states besides the most recent
            if early_transitions || has_fast_guarantee {
                for (st, g) in &prev_states {
                    let early_go = build_assignments!(self.builder;
                        fsm["start"] = g ? signal_on["out"];
                    );
                    self.fsm_enables.entry(*st).or_default().extend(early_go);
                }
            }

            let transitions = prev_states
                .into_iter()
                .map(|(st, guard)| (st, cur_state, guard));
            self.transitions.extend(transitions);

            let done_cond = guard!(fsm["done"]);
            Ok(vec![(cur_state, done_cond)])

        },
        // See explanation of FSM states generated in [ir::TopDownCompileControl].
        ir::Control::Enable(ir::Enable { group, attributes }) => {
            let cur_state = attributes.get(NODE_ID).unwrap_or_else(|| panic!("Group `{}` does not have state_id information", group.borrow().name()));
            log::info!("cur stat: {}, group: {}", attributes.get(NODE_ID).unwrap()+1, group.borrow().name() );
            let (cur_state, prev_states) = (cur_state, preds);

            // Add group to mapping for emitting group JSON info
            self.groups_to_states.insert(FSMStateInfo { id: cur_state, group: group.borrow().name() });

            let done_cond = group.borrow().done_cond().clone();

            let group_done_cond =  if done_cond.src.borrow().is_constant(){
                ir::Guard::True
            } else {
                ir::Guard::Not(Box::new(ir::Guard::Port(done_cond.src)))
            };
            
            let signal_on = self.builder.add_constant(1, 1);

            // Activate this group in the current state
            let assigns: Vec<Assignment<Nothing>> = group.borrow_mut().assignments.clone();
            for assign in assigns.iter(){
                if assign.dst.borrow().name == "done"{
                    continue;
                }
                self.enables.entry(cur_state).or_default().push(ir::Assignment{
                    src: assign.src.clone(),
                    dst: assign.dst.clone(),
                    attributes: assign.attributes.clone(),
                    guard: Box::new(group_done_cond.clone()),
                });
            }
 
            // Activate group in the cycle when previous state signals done.
            // NOTE: We explicilty do not add `not_done` to the guard.
            // See explanation in [ir::TopDownCompileControl] to understand
            // why.
            if early_transitions || has_fast_guarantee {
                for (st, g) in &prev_states {
                    let early_go = build_assignments!(self.builder;
                        group["go"] = g ? signal_on["out"];
                    );
                    self.enables.entry(*st).or_default().extend(early_go);
                }
            }

            let transitions = prev_states
                .into_iter()
                .map(|(st, guard)| (st, cur_state, guard));
            self.transitions.extend(transitions);
            let done_cond = group.borrow().done_cond().clone();
            let done_cond_guard = ir::Guard::And(done_cond.guard, Box::new(ir::Guard::Port(done_cond.src))).simplify();
            Ok(vec![(cur_state, done_cond_guard)])
        }
        ir::Control::Seq(seq) => {
            self.calc_seq_recur(seq, preds, early_transitions)
        }
        ir::Control::If(if_stmt) => {
            self.calc_if_recur(if_stmt, preds, early_transitions)
        }
        ir::Control::While(while_stmt) => {
            self.calc_while_recur(while_stmt, preds, early_transitions)
        }
        ir::Control::Par(_) => unreachable!(),
        ir::Control::Repeat(_) => unreachable!("`repeat` statements should have been compiled away. Run `{}` before this pass.", passes::CompileRepeat::name()),
        ir::Control::Invoke(_) => unreachable!("`invoke` statements should have been compiled away. Run `{}` before this pass.", passes::CompileInvoke::name()),
        ir::Control::Empty(_) => unreachable!("`calculate_states_recur` should not see an `empty` control."),
        ir::Control::Static(sc) => {
            if let ir::StaticControl::Enable(ir::StaticEnable{group, attributes }) = sc {
                log::info!("cur stat: {}, group: {}, latency: {}", attributes.get(NODE_ID).unwrap()+1, group.borrow().name(), sc.get_latency());
                
                let cur_state = attributes.get(NODE_ID).unwrap_or_else(
                    || panic!("Group `{}` does not have state_id information", group.borrow().name())
                );

                let (cur_state, prev_states) = if preds.len() == 1 && preds[0].1.is_true() && preds[0].0 == 0 {
                    (preds[0].0, vec![])
                } else {
                    (cur_state, preds)
                };

                // Add group to mapping for emitting group JSON info
                self.groups_to_states.insert(FSMStateInfo { id: cur_state, group: group.borrow().name() });

                let assigns: Vec<Assignment<StaticTiming>> = group.borrow_mut().assignments.clone();
                for assign in assigns.iter(){
                    if let Some(timing_interval) =  assign.guard.get_timing_interval() {
                        let (u, v) = timing_interval;
                        // Convert the guard for each iteration
                        for i in u..v{
                            let new_guard = assign.guard.clone().replace_static_timing_at_time(u).simplify();
                            self.enables
                                .entry(cur_state+i)
                                .or_default()
                                .push(ir::Assignment{
                                    src: assign.src.clone(),
                                    dst: assign.dst.clone(),
                                    attributes: assign.attributes.clone(),
                                    guard: Box::new(ir::Guard::<Nothing>::from(new_guard)),
                                });
                        }
                    }
                    else {
                        for i in 0..sc.get_latency() {
                        // Convert the guard for each iteration
                        self.enables
                            .entry(cur_state+i)
                            .or_default()
                            .push(ir::Assignment{
                                src: assign.src.clone(),
                                dst: assign.dst.clone(),
                                attributes: assign.attributes.clone(),
                                guard: Box::new(ir::Guard::True),
                            });
                        }
                    }
                }

                let mut transitions = prev_states
                    .into_iter()
                    .map(|(st, guard)| {
                            (st, cur_state, guard)
                        }
                    ).collect_vec();

                for i in cur_state..cur_state + sc.get_latency()  {
                    transitions.push((i, i + 1, ir::Guard::True));
                }
                    
                for t in transitions.into_iter(){
                    if self.transitions.contains(&t){
                        continue;
                    }
                    else{
                        self.transitions.push(t);
                    }
                }
                
                // always transition to the next state
                let done_cond = ir::Guard::True;
                Ok(vec![(cur_state + sc.get_latency(), done_cond)])
            }else{
                unreachable!("`calculate_states_recur` should not see a static control that is not an enable.")
            }
        }
    }
    }

    /// Builds a finite state machine for `seq` represented by a [Schedule].
    /// At a high level, it iterates through each stmt in the seq's control, using the
    /// previous stmt's [PredEdge] as the `preds` for the current stmt, and returns
    /// the [PredEdge] implied by the last stmt in `seq`'s control.
    fn calc_seq_recur(
        &mut self,
        seq: &ir::Seq,
        // The set of previous states that want to transition into cur_state
        preds: Vec<PredEdge>,
        // True if early_transitions are allowed
        early_transitions: bool,
    ) -> CalyxResult<Vec<PredEdge>> {
        let mut prev = preds;
        for (i, stmt) in seq.stmts.iter().enumerate() {
            prev = self.calculate_states_recur(
                stmt,
                prev,
                early_transitions,
                i > 0 && seq.get_attributes().has(BoolAttr::Fast),
            )?;
        }
        Ok(prev)
    }

    /// Builds a finite state machine for `if_stmt` represented by a [Schedule].
    /// First generates the transitions into the true branch + the transitions that exist
    /// inside the true branch. Then generates the transitions into the false branch + the transitions
    /// that exist inside the false branch. Then calculates the transitions needed to
    /// exit the if statmement (which include edges from both the true and false branches).
    fn calc_if_recur(
        &mut self,
        if_stmt: &ir::If,
        // The set of previous states that want to transition into cur_state
        preds: Vec<PredEdge>,
        // True if early_transitions are allowed
        early_transitions: bool,
    ) -> CalyxResult<Vec<PredEdge>> {
        if if_stmt.cond.is_some() {
            return Err(Error::malformed_structure(format!("{}: Found group `{}` in with position of if. This should have compiled away.", DynamicFSMAllocation::name(), if_stmt.cond.as_ref().unwrap().borrow().name())));
        }
        let port_guard: ir::Guard<Nothing> = Rc::clone(&if_stmt.port).into();
        // Previous states transitioning into true branch need the conditional
        // to be true.
        let tru_transitions = preds
            .clone()
            .into_iter()
            .map(|(s, g)| (s, g & port_guard.clone()))
            .collect();
        let tru_prev = self.calculate_states_recur(
            &if_stmt.tbranch,
            tru_transitions,
            early_transitions,
            false,
        )?;
        // Previous states transitioning into false branch need the conditional
        // to be false.
        let fal_transitions = preds
            .into_iter()
            .map(|(s, g)| (s, g & !port_guard.clone()))
            .collect();

        let fal_prev = if let ir::Control::Empty(..) = *if_stmt.fbranch {
            // If the false branch is empty, then all the prevs to this node will become prevs
            // to the next node.
            fal_transitions
        } else {
            self.calculate_states_recur(
                &if_stmt.fbranch,
                fal_transitions,
                early_transitions,
                false,
            )?
        };

        let prevs = tru_prev.into_iter().chain(fal_prev).collect();
        Ok(prevs)
    }

    /// Builds a finite state machine for `while_stmt` represented by a [Schedule].
    /// It first generates the backwards edges (i.e., edges from the end of the while
    /// body back to the beginning of the while body), then generates the forwards
    /// edges in the body, then generates the edges that exit the while loop.
    fn calc_while_recur(
        &mut self,
        while_stmt: &ir::While,
        // The set of previous states that want to transition into cur_state
        preds: Vec<PredEdge>,
        // True if early_transitions are allowed
        early_transitions: bool,
    ) -> CalyxResult<Vec<PredEdge>> {
        if while_stmt.cond.is_some() {
            return Err(Error::malformed_structure(format!("{}: Found group `{}` in with position of if. This should have compiled away.", DynamicFSMAllocation::name(), while_stmt.cond.as_ref().unwrap().borrow().name())));
        }

        let port_guard: ir::Guard<Nothing> = Rc::clone(&while_stmt.port).into();

        // Step 1: Generate the backward edges by computing the exit nodes.
        let mut exits = vec![];
        control_exits(&while_stmt.body, &mut exits);

        // Step 2: Generate the forward edges normally.
        // Previous transitions into the body require the condition to be
        // true.
        let transitions: Vec<PredEdge> = preds
            .clone()
            .into_iter()
            .chain(exits)
            .map(|(s, g)| (s, g & port_guard.clone()))
            .collect();
        let prevs = self.calculate_states_recur(
            &while_stmt.body,
            transitions,
            early_transitions,
            false,
        )?;

        // Step 3: The final out edges from the while come from:
        //   - Before the body when the condition is false
        //   - Inside the body when the condition is false
        let not_port_guard = !port_guard;
        let all_prevs: Vec<PredEdge> = preds
            .into_iter()
            .chain(prevs)
            .map(|(st, guard)| (st, guard & not_port_guard.clone()))
            .collect();

        Ok(all_prevs)
    }

    /// Creates a Schedule that represents `seq`, mainly relying on `calc_seq_recur()`.
    fn calculate_states_seq(
        &mut self,
        seq: &ir::Seq,
        early_transitions: bool,
    ) -> CalyxResult<()> {
        let first_state = (0, ir::Guard::True);
        // We create an empty first state in case the control program starts with
        // a branch (if, while).
        // If the program doesn't branch, then the initial state is merged into
        // the first group.
        let prev =
            self.calc_seq_recur(seq, vec![first_state], early_transitions)?;
        self.add_nxt_transition(prev);
        Ok(())
    }

    fn calculate_states_while(
        &mut self,
        while_stmt: &ir::While,
        early_transitions: bool,
    ) -> CalyxResult<()> {
        let first_state = (0, ir::Guard::True);
        let prev =
            self.calc_while_recur(while_stmt, vec![first_state], early_transitions)?;
        self.add_nxt_transition(prev);
        Ok(())
    }

    fn calculate_states_if(
        &mut self,
        if_stmt: &ir::If,
        early_transitions: bool,
    ) -> CalyxResult<()> {
        let first_state = (0, ir::Guard::True);
        let prev =
            self.calc_if_recur(if_stmt, vec![first_state], early_transitions)?;
        self.add_nxt_transition(prev);
        Ok(())
    }

    /// Given predecessors prev, creates a new "next" state and transitions from
    /// each state in prev to the next state.
    /// In other words, it just adds an "end" state to [Schedule] and the
    /// appropriate transitions to that "end" state.
    fn add_nxt_transition(&mut self, prev: Vec<PredEdge>) {
        let nxt = prev
            .iter()
            .max_by(|(st1, _), (st2, _)| st1.cmp(st2))
            .unwrap()
            .0
            + 1;
        let transitions = prev.into_iter().map(|(st, guard)| (st, nxt, guard));
        self.transitions.extend(transitions);
    }

    /// Note: the functions calculate_states_seq, calculate_states_while, and calculate_states_if
    /// are functions that basically do what `calculate_states` would do if `calculate_states` knew (for certain)
    /// that its input parameter would be a seq/while/if.
    /// The reason why we need to define these as separate functions is because `finish_seq`
    /// (for example) we only gives us access to a `& mut seq` type, not a `& Control`
    /// type.
    fn calculate_states(
        &mut self,
        con: &ir::Control,
        early_transitions: bool,
    ) -> CalyxResult<()> {
        // let first_state = (0, ir::Guard::True, false);
        // We create an empty first state in case the control program starts with
        // a branch (if, while).
        // If the program doesn't branch, then the initial state is merged into
        // the first group.
        let prev =
            self.calculate_states_recur(con, vec![], early_transitions, false)?;
        self.add_nxt_transition(prev);
        Ok(())
    }
}

/// **Core lowering pass.**
/// Compiles away the control programs in components into purely structural code using an
/// finite-state machine (FSM).
///
/// Lowering operates in two steps:
/// 1. Compile all [ir::Par] control sub-programs into a single [ir::Enable] of a group that runs
///    all children to completion.
/// 2. Compile the top-level control program into a single [ir::Enable].
///
/// ## Compiling non-`par` programs
/// At very high-level, the pass assigns an FSM state to each [ir::Enable] in the program and
/// generates transitions to the state to activate the groups contained within the [ir::Enable].
///
/// The compilation process calculates all predeccesors of the [ir::Enable] while walking over the
/// control program. A predeccesor is any enable statement that can directly "jump" to the current
/// [ir::Enable]. The compilation process computes all such predeccesors and the guards that need
/// to be true for the predeccesor to jump into this enable statement.
///
/// ```
/// cond0;
/// while lt.out {
///   if gt.out { true } else { false }
/// }
/// next;
/// ```
/// The predeccesor sets are:
/// ```
/// cond0 -> []
/// true -> [(cond0, lt.out & gt.out); (true; lt.out & gt.out); (false, lt.out & !gt.out)]
/// false -> [(cond0, lt.out & !gt.out); (true; lt.out & gt.out); (false, lt.out & !gt.out)]
/// next -> [(cond0, !lt.out); (true, !lt.out); (false, !lt.out)]
/// ```
///
/// ### Compiling [ir::Enable]
/// The process first takes all edges from predeccesors and transitions to the state for this
/// enable and enables the group in this state:
/// ```text
/// let cur_state; // state of this enable
/// for (state, guard) in predeccesors:
///   transitions.insert(state, cur_state, guard)
/// enables.insert(cur_state, group)
/// ```
///
/// While this process will generate a functioning FSM, the FSM takes unnecessary cycles for FSM
/// transitions.
///
/// For example:
/// ```
/// seq { one; two; }
/// ```
/// The FSM generated will look like this (where `f` is the FSM register):
/// ```
/// f.in = one[done] ? 1;
/// f.in = two[done] ? 2;
/// one[go] = !one[done] & f.out == 0;
/// two[go] = !two[done] & f.out == 1;
/// ```
///
/// The cycle-level timing for this FSM will look like:
///     - cycle 0: (`f.out` == 0), enable one
///     - cycle t: (`f.out` == 0), (`one[done]` == 1), disable one
///     - cycle t+1: (`f.out` == 1), enable two
///     - cycle t+l: (`f.out` == 1), (`two[done]` == 1), disable two
///     - cycle t+l+1: finish
///
/// The transition t -> t+1 represents one where group one is done but group two hasn't started
/// executing.
///
/// To address this specific problem, there is an additional enable added to run all groups within
/// an enable *while the FSM is transitioning*.
/// The final transition will look like this:
/// ```
/// f.in = one[done] ? 1;
/// f.in = two[done] ? 2;
/// one[go] = !one[done] & f.out == 0;
/// two[go] = (!two[done] & f.out == 1) || (one[done] & f.out == 0);
/// ```
///
/// Note that `!two[done]` isn't present in the second disjunct because all groups are guaranteed
/// to run for at least one cycle and the second disjunct will only be true for one cycle before
/// the first disjunct becomes true.
///
/// ## Compiling `par` programs
/// We have to generate new FSM-based controller for each child of a `par` node so that each child
/// can indepdendently make progress.
/// If we tie the children to one top-level FSM, their transitions would become interdependent and
/// reduce available concurrency.
///
/// ## Compilation guarantee
/// At the end of this pass, the control program will have no more than one
/// group enable in it.
pub struct DynamicFSMAllocation {
    /// Print out the FSM representation to STDOUT
    dump_fsm: bool,
    /// Print out the FSM representation to a dot file
    dump_dot: String,
    /// Enable early transitions
    early_transitions: bool,
    /// Bookkeeping for FSM ids for groups across all FSMs in the program
    fsm_groups: HashSet<ProfilingInfo>,
}

impl ConstructVisitor for DynamicFSMAllocation {
    fn from(ctx: &ir::Context) -> CalyxResult<Self>
    where
        Self: Sized + Named,
    {
        let opts = Self::get_opts(ctx);

        Ok(DynamicFSMAllocation {
            dump_fsm: opts[&"dump-fsm"].bool(),
            dump_dot: opts[&"dump-dot"].string(),
            early_transitions: opts[&"early-transitions"].bool(),
            fsm_groups: HashSet::new(),
        })
    }

    fn clear_data(&mut self) {
        /* All data can be transferred between components */
    }
}

impl Named for DynamicFSMAllocation {
    fn name() -> &'static str {
        "dfsm"
    }

    fn description() -> &'static str {
        "Removing control constructs and instantiate FSMs in their place"
    }

    fn opts() -> Vec<PassOpt> {
        vec![
            PassOpt::new(
                "dump-fsm",
                "Print out the state machine implementing the schedule",
                ParseVal::Bool(false),
                PassOpt::parse_bool,
            ),
            PassOpt::new(
                "dump-dot",
                "Print out the state machine implementing the schedule to a dot file",
                ParseVal::String("".to_string()),
                PassOpt::parse_string,
            ),
            PassOpt::new(
                "dump-fsm-json",
                "Write the state machine implementing the schedule to a JSON file",
                ParseVal::OutStream(OutputFile::Null),
                PassOpt::parse_outstream,
            ),
            PassOpt::new(
                "early-transitions",
                "Experimental: Enable early transitions for group enables",
                ParseVal::Bool(false),
                PassOpt::parse_bool,
            ),
        ]
    }
}

/// Helper function to emit profiling information when the control consists of a single group.
fn extract_single_enable(
    con: &mut ir::Control,
    component: Id,
) -> Option<SingleEnableInfo> {
    if let ir::Control::Enable(enable) = con {
        return Some(SingleEnableInfo {
            component,
            group: enable.group.borrow().name(),
        });
    } else {
        None
    }
}

impl Visitor for DynamicFSMAllocation {
    fn start(
        &mut self,
        comp: &mut calyx_ir::Component,
        _sigs: &LibrarySignatures,
        _comps: &[calyx_ir::Component],
    ) -> VisResult {
        let mut con = comp.control.borrow_mut();
        if matches!(*con, ir::Control::Empty(..) | ir::Control::Enable(..)) {
            if let Some(enable_info) =
                extract_single_enable(&mut con, comp.name)
            {
                self.fsm_groups
                    .insert(ProfilingInfo::SingleEnable(enable_info));
            }
            return Ok(Action::Stop);
        }
        Ok(Action::Continue)
    }

    fn finish_seq(
        &mut self,
        s: &mut calyx_ir::Seq,
        comp: &mut calyx_ir::Component,
        sigs: &LibrarySignatures,
        _comps: &[calyx_ir::Component],
    ) -> VisResult {
        if !s.attributes.has(ir::BoolAttr::NewFSM) {
            return Ok(Action::Continue);
        }
        let mut builder = ir::Builder::new(comp, sigs);
        let mut sch = Schedule::from(&mut builder);
        sch.calculate_states_seq(s, self.early_transitions)?;
        let seq_fsm = sch.realize_fsm(self.dump_fsm, &self.dump_dot);
        let mut fsm_en = ir::Control::fsm_enable(seq_fsm);
        let state_id = s.attributes.get(NODE_ID).unwrap();
        fsm_en.get_mut_attributes().insert(NODE_ID, state_id);
        Ok(Action::change(fsm_en))
    }

    fn finish_par(
        &mut self,
        s: &mut calyx_ir::Par,
        comp: &mut calyx_ir::Component,
        sigs: &LibrarySignatures,
        _comps: &[calyx_ir::Component],
    ) -> VisResult {
        let mut builder = ir::Builder::new(comp, sigs);

        // Compilation FSM
        let mut assigns_to_enable = vec![];

        structure!(builder;
            let signal_on = constant(1, 1);
        );

        // Registers to save the done signal from each child.
        let mut done_regs: Vec<RRC<ir::Cell>> =
            Vec::with_capacity(s.stmts.len());

        // replace every thread with a single-element sequential schedule
        // in order to instantiate a 3-state FSM for the thread
        let threads = s
            .stmts
            .drain(..)
            .map(|s| ir::Control::seq(vec![s]))
            .collect_vec();

        s.stmts.extend(threads);

        // For each child, build an FSM to run the thread
        for con in &s.stmts {
            // regardless of the type of thread (seq / enable / etc.),
            // instantiate an FSM; we might want to change this in the future
            // to enable no transition latency between par-thread-go and
            // when the thread actually begins working (the common case might
            // simply be a group, which would mean a 1-cycle group takes 3 cycles now)
            let mut sch = Schedule::from(&mut builder);
            sch.calculate_states(con, self.early_transitions)?;
            let fsm = sch.realize_fsm(self.dump_fsm, &self.dump_dot);

            // Build circuitry to enable and disable this fsm.
            structure!(builder;
                let pd = prim std_reg(1);
            );

            let fsm_go = !(guard!(pd["out"] | fsm["done"]));
            let fsm_done = guard!(fsm["done"]);

            // save the go / done assignments for fsm representing the par thread
            let par_thread_assigns: [ir::Assignment<Nothing>; 3] = build_assignments!(builder;
                fsm["start"] = fsm_go ? signal_on["out"];
                pd["in"] = fsm_done ? signal_on["out"];
                pd["write_en"] = fsm_done ? signal_on["out"];
            );

            assigns_to_enable.extend(par_thread_assigns);
            done_regs.push(pd)
        }

        // Done condition for par block's FSM
        let true_guard = ir::Guard::True;
        let par_fsm = builder.add_fsm("par");
        let transition_to_done: ir::Guard<Nothing> = done_regs
            .clone()
            .into_iter()
            .map(|r| guard!(r["out"]))
            .fold(true_guard.clone(), ir::Guard::and);

        // generate transition conditions for each state of the par's FSM
        let par_fsm_trans = vec![
            // conditional transition from IDLE to COMPUTE on par_fsm_start
            ir::Transition::Conditional(vec![
                (guard!(par_fsm["start"]), 1),
                (true_guard.clone(), 0),
            ]),
            // conditional transition from COMPUTE to DONE based on completion of all done regs
            ir::Transition::Conditional(vec![
                (transition_to_done, 2),
                (true_guard.clone(), 1),
            ]),
            ir::Transition::Unconditional(0),
        ];

        // generate assignments to occur at each state of par's FSM
        let par_fsm_assigns = vec![
            vec![],
            assigns_to_enable,
            build_assignments!(builder;
                par_fsm["done"] = true_guard ? signal_on["out"];
            )
            .to_vec(),
        ];

        // place all of these into the FSM
        par_fsm.borrow_mut().assignments.extend(par_fsm_assigns);
        par_fsm.borrow_mut().transitions.extend(par_fsm_trans);

        // put the state id of the par schedule onto the par fsm
        let mut en = ir::Control::fsm_enable(par_fsm);
        let state_id = s.attributes.get(NODE_ID).unwrap();
        en.get_mut_attributes().insert(NODE_ID, state_id);        
        Ok(Action::change(en))
    }

    fn finish_while(
        &mut self,
        w: &mut calyx_ir::While,
        comp: &mut calyx_ir::Component,
        sigs: &LibrarySignatures,
        _comps: &[calyx_ir::Component],
    ) -> VisResult {
        if !w.attributes.has(ir::BoolAttr::NewFSM) {
            return Ok(Action::Continue);
        }
        let mut builder = ir::Builder::new(comp, sigs);
        let mut sch = Schedule::from(&mut builder);
        sch.calculate_states_while(w, self.early_transitions)?;
        let fsm = sch.realize_fsm(self.dump_fsm, &self.dump_dot);
        let mut fsm_en = ir::Control::fsm_enable(fsm);
        let state_id = w.attributes.get(NODE_ID).unwrap();
        fsm_en.get_mut_attributes().insert(NODE_ID, state_id);
        Ok(Action::change(fsm_en))
    }
    
    fn finish_if(
        &mut self,
        i: &mut calyx_ir::If,
        comp: &mut calyx_ir::Component,
        sigs: &LibrarySignatures,
        _comps: &[calyx_ir::Component],
    ) -> VisResult {
        if !i.attributes.has(ir::BoolAttr::NewFSM) {
            return Ok(Action::Continue);
        }
        let mut builder = ir::Builder::new(comp, sigs);
        let mut sch = Schedule::from(&mut builder);
        sch.calculate_states_if(i, self.early_transitions)?;
        let fsm = sch.realize_fsm(self.dump_fsm, &self.dump_dot);
        let mut fsm_en = ir::Control::fsm_enable(fsm);
        let state_id = i.attributes.get(NODE_ID).unwrap();
        fsm_en.get_mut_attributes().insert(NODE_ID, state_id);
        Ok(Action::change(fsm_en))
    }
    
    fn finish(
        &mut self,
        comp: &mut ir::Component,
        sigs: &LibrarySignatures,
        _comps: &[ir::Component],
    ) -> VisResult {
        let control = Rc::clone(&comp.control);
        
        let mut builder = ir::Builder::new(comp, sigs);
        let mut sch = Schedule::from(&mut builder);
        
        // Add assignments for the final states
        sch.calculate_states(&control.borrow(), self.early_transitions)?;
        let comp_fsm = sch.realize_fsm(self.dump_fsm, &self.dump_dot);
        Ok(Action::change(ir::Control::fsm_enable(comp_fsm)))
    }
}
