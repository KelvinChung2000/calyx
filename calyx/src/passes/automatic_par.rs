use crate::lang::component::Component;
use crate::lang::{ast, ast::Control, context::Context};
use crate::passes::visitor::{Action, VisResult, Visitor};

/// Pass that collapses
/// (seq
/// (seq (enable A B)
/// (enable C D))
/// ..)
/// into
/// (seq (enable A B C D)
/// ..)
/// given that there are no edges between the sub-graphs induced by (enable A B) and (enable C D)
/// since in this case there is no way for these subgraphs to depend on each other
/// XXX (zhijing): I think this pass need to be changed if we add `enable` CSP style components to futil semantics
///
/// For example, suppose that this were the structure graph of your component:
/// ```
/// ╭─╮    ╭─╮
/// │A│    │C│
/// ╰┬╯    ╰┬╯
///  │      │
///  │      │
///  v      v
/// ╭─╮    ╭─╮
/// │B│    │D│
/// ╰─╯    ╰─╯
/// ```
/// In this case, the program
/// ```
/// (seq (enable A B) (enable C D))
/// ```
/// is equivalent to
/// ```
/// (seq (enable A B C D))
/// ```
/// because there are no edges between the sub-graph induced by `A` and `B`
/// and the sub-graph induced by `C` and `D`.
///
/// If instead this were your component graph:
/// ```
/// ╭─╮    ╭─╮
/// │A│───>│C│
/// ╰┬╯    ╰┬╯
///  │      │
///  │      │
///  v      v
/// ╭─╮    ╭─╮
/// │B│    │D│
/// ╰─╯    ╰─╯
/// ```
/// then you could not collapse the `seq`.
#[derive(Default)]
pub struct AutomaticPar {}

impl Visitor for AutomaticPar {
    fn name(&self) -> String {
        "automatic parallelization".to_string()
    }

    // use finish_seq so that we collapse things on the way
    // back up the tree and potentially catch more cases
    fn finish_seq(
        &mut self,
        seq: &mut ast::Seq,
        comp: &mut Component,
        _: &Context,
    ) -> VisResult {
        let st = &mut comp.structure;

        // get first enable statement and its index. this will act
        // as the accumulator for the statement that we are collapsing
        // things into
        let (start, mut cmp_acc) = match seq.stmts.iter().enumerate().find_map(
            |(i, stmt)| match stmt {
                Control::Enable { data } => Some((i, data.clone())),
                _ => None,
            },
        ) {
            Some((s, c)) => (s, c),
            None => return Ok(Action::Continue),
        };

        // vec of new control statements including all elements up to the first
        // enable that we find
        let mut new_stmts: Vec<ast::Control> = seq.stmts[..start].to_vec();

        // start interation from the start+1 because the start
        // is in `cmp_stmt`
        for stmt in seq.stmts[start + 1..].iter() {
            match stmt {
                Control::Enable { data: enables2 } => {
                    let mut conflict = false;
                    // for every component in the `enables2` we check if any
                    // incoming/outgoing edge has an endpoint in `cmp_acc`
                    for en_comp in &enables2.comps {
                        let idx = st.get_inst_index(en_comp)?;
                        // for every output port, check if any incoming/outgoing edges
                        // contain a component in `cmp_acc`
                        conflict |= st.graph[idx].out_ports().any(|port| {
                            let outgoing = st
                                .connected_to(idx, port.to_string())
                                .any(|(node_data, _)| {
                                    cmp_acc.comps.contains(node_data.get_name())
                                });
                            let incoming = st
                                .connected_from(idx, port.to_string())
                                .any(|(node_data, _)| {
                                    cmp_acc.comps.contains(node_data.get_name())
                                });
                            outgoing || incoming
                        });
                    }
                    if conflict {
                        new_stmts.push(Control::enable(cmp_acc.comps.clone()));
                        // there was a conflict, set this to the new accumulator
                        cmp_acc = enables2.clone();
                    } else {
                        cmp_acc.comps.append(&mut enables2.comps.clone());
                    }
                }
                _ => new_stmts.push(stmt.clone()),
            }
        }
        new_stmts.push(Control::enable(cmp_acc.comps));
        Ok(Action::Change(Control::seq(new_stmts)))
    }
}
