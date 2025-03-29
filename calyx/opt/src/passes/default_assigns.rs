use crate::analysis::AssignmentAnalysis;
use crate::traversal::{Action, ConstructVisitor, Named, VisResult, Visitor};
use calyx_ir::{self as ir, LibrarySignatures};
use calyx_utils::{CalyxResult, Error};
use itertools::Itertools;
use std::collections::HashMap;

/// Adds default assignments to all non-`@data` ports of an instance.
pub struct DefaultAssigns {
    /// Mapping from component to data ports
    data_ports: HashMap<ir::Id, Vec<ir::Id>>,
}

impl Named for DefaultAssigns {
    fn name() -> &'static str {
        "default-assigns"
    }

    fn description() -> &'static str {
        "adds default assignments to all non-`@data` ports of an instance."
    }
}

impl ConstructVisitor for DefaultAssigns {
    fn from(ctx: &ir::Context) -> CalyxResult<Self>
    where
        Self: Sized,
    {
        let data_ports = ctx
            .lib
            .signatures()
            .map(|sig| {
                let ports = sig.signature.iter().filter_map(|p| {
                    if p.direction == ir::Direction::Input
                        && !p.attributes.has(ir::BoolAttr::Data)
                        && !p.attributes.has(ir::BoolAttr::Clk)
                        && !p.attributes.has(ir::BoolAttr::Reset)
                    {
                        Some(p.name())
                    } else {
                        None
                    }
                });
                (sig.name, ports.collect())
            })
            .collect();
        Ok(Self { data_ports })
    }

    fn clear_data(&mut self) {
        /* shared across components */
    }
}

impl Visitor for DefaultAssigns {
    fn start(
        &mut self,
        comp: &mut ir::Component,
        sigs: &LibrarySignatures,
        _comps: &[ir::Component],
    ) -> VisResult {
        if !comp.is_structural() {
            return Err(Error::pass_assumption(
                Self::name(),
                format!("component {} is not purely structural", comp.name),
            ));
        }

        // We only need to consider write set of the continuous assignments
        let mut writes = comp
            .continuous_assignments
            .iter()
            .analysis()
            .writes()
            .group_by_cell();

        comp.fsms.iter().for_each(|fsm| {
            fsm.borrow().assignments.iter().for_each(|assigns| {
                for (k, v) in assigns.iter().analysis().writes().group_by_cell().iter(){
                    for i in v.iter(){
                        if writes.entry(*k).or_default().contains(i){
                            continue;
                        }
                        writes.entry(*k).or_default().push(i.clone());
                    }
                }
            });
        });

        let mut assigns = Vec::new();

        let mt = vec![];
        let cells = comp.cells.iter().cloned().collect_vec();
        let ports = comp.signature.borrow().ports().iter().cloned().collect_vec();
        let mut con_assigns = comp.continuous_assignments.iter().cloned().collect_vec();
        comp.fsms.iter().for_each(|fsm| {
            fsm.borrow().assignments.iter().for_each(|assigns| {
                con_assigns
                    .extend(assigns.iter().cloned().collect_vec());
            })
        });

        let mut builder = ir::Builder::new(comp, sigs);

        for cr in &cells {
            let cell = cr.borrow();
            let Some(typ) = cell.type_name() else {
                continue;
            };
            let Some(required) = self.data_ports.get(&typ) else {
                continue;
            };

            // For all the assignments not in the write set, add a default assignment
            // if the assignment does not write to an FSM-controlling register
            let mut cell_writes: Vec<ir::RRC<ir::Port>> = writes
                .get(&cell.name())
                .unwrap_or(&mt)
                .iter()
                .map(ir::RRC::clone)
                .collect();

            if cell.attributes.has(ir::BoolAttr::FSMControl) {
                cell_writes
                    .extend(cell.ports().into_iter().map(ir::RRC::clone));
            }

            let cell_writes = cell_writes
                .into_iter()
                .map(|p| {
                    let p = p.borrow();
                    p.name
                })
                .collect_vec();

            assigns.extend(
                required.iter().filter(|p| !cell_writes.contains(p)).map(
                    |name| {
                        let port = cell.get(name);
                        let zero = builder.add_constant(0, port.borrow().width);
                        let assign: ir::Assignment<ir::Nothing> = builder
                            .build_assignment(
                                cell.get(name),
                                zero.borrow().get("out"),
                                ir::Guard::True,
                            );
                        log::info!(
                            "Adding default assignment for {}",
                            ir::Printer::assignment_to_str(&assign)
                        );
                        assign
                    },
                ),
            );
        }

        for port in &ports {
            if port.borrow().direction == ir::Direction::Output {
                continue;
            }
            let port_name = port.borrow().name.clone();
            // Check if any existing assignment writes to this port
            if !con_assigns.iter().any(|assign| {
                assign.dst.borrow().name == port_name
            }) {
                let zero = builder.add_constant(0, port.borrow().width);
                let assign: ir::Assignment<ir::Nothing> = builder
                    .build_assignment(
                        port.clone(),
                        zero.borrow().get("out"),
                        ir::Guard::True,
                    );
                log::info!(
                    "Adding default assignment for port {}",
                    port_name
                );
                assigns.push(assign);
            }
        }

        comp.continuous_assignments.extend(assigns);

        // Purely structural pass
        Ok(Action::Stop)
    }
}
