use std::{
    cmp::{max, Ordering},
    collections::HashMap,
    hash::Hash,
    num::NonZeroU32,
};

use crate::flatten::{
    flat_ir::base::GlobalCellIdx,
    structures::{
        index_trait::{impl_index_nonzero, IndexRef},
        indexed_map::IndexedMap,
        thread::ThreadIdx,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockIdx(NonZeroU32);
impl_index_nonzero!(ClockIdx);

use baa::BitVecValue;
use itertools::Itertools;
use thiserror::Error;

pub type ClockMap = IndexedMap<ClockIdx, VectorClock<ThreadIdx>>;
pub type ThreadClockPair = (ThreadIdx, ClockIdx);

impl ClockMap {
    /// pushes a new clock into the map and returns its index
    pub fn new_clock(&mut self) -> ClockIdx {
        self.push(VectorClock::new())
    }

    /// Returns a new clock that is the clone of the given clock
    pub fn fork_clock(&mut self, parent: ClockIdx) -> ClockIdx {
        self.push(self[parent].clone())
    }
}

pub trait Counter: Default {
    /// Increment the counter, returning `None` if the counter overflowed.
    #[must_use]
    fn increment(&mut self) -> Option<()>;

    /// Increment the counter, panicking if the counter overflowed.
    fn increment_expect(&mut self) {
        self.increment().expect("counter overflowed");
    }
}

impl Counter for u8 {
    fn increment(&mut self) -> Option<()> {
        *self = self.checked_add(1)?;
        Some(())
    }
}

impl Counter for u16 {
    fn increment(&mut self) -> Option<()> {
        *self = self.checked_add(1)?;
        Some(())
    }
}

impl Counter for u32 {
    fn increment(&mut self) -> Option<()> {
        *self = self.checked_add(1)?;
        Some(())
    }
}

impl Counter for u64 {
    fn increment(&mut self) -> Option<()> {
        *self = self.checked_add(1)?;
        Some(())
    }
}

// I don't expect this to be used much, but it's here for completeness
impl Counter for u128 {
    fn increment(&mut self) -> Option<()> {
        *self = self.checked_add(1)?;
        Some(())
    }
}

/// If the clock map is provided, use it to create a new clock. Otherwise,
/// return the 0th clock idx.
pub fn new_clock(clock_map: &mut Option<&mut ClockMap>) -> ClockIdx {
    clock_map
        .as_mut()
        .map(|c| c.new_clock())
        .unwrap_or(ClockIdx::new(0))
}

/// A simple vector clock implementation.
///
/// Internally uses a [`HashMap`] to store the clock values. Keys which are not
/// present in the map are assumed to be the default value for the given counter
/// type, which is zero for the standard integer counters. This means that all
/// threads implicitly start at zero, rather than some bottom value.
#[derive(Debug, Clone)]
pub struct VectorClock<I, C = u32>
where
    I: Hash + Eq + Clone,
    C: Ord + Clone + Counter,
{
    // TODO: maybe use `ahash` instead
    map: HashMap<I, C>,
}

impl<I, C> std::ops::Index<&I> for VectorClock<I, C>
where
    I: Hash + Eq + Clone,
    C: Ord + Clone + Counter,
{
    type Output = C;

    fn index(&self, index: &I) -> &Self::Output {
        &self.map[index]
    }
}

impl<I, C> std::ops::Index<I> for VectorClock<I, C>
where
    I: Hash + Eq + Clone,
    C: Ord + Clone + Counter,
{
    type Output = C;

    fn index(&self, index: I) -> &Self::Output {
        &self.map[&index]
    }
}

impl<I, C> Eq for VectorClock<I, C>
where
    I: Hash + Eq + Clone,
    C: Ord + Clone + Counter,
{
}

impl<I, C> PartialEq for VectorClock<I, C>
where
    I: Hash + Eq + Clone,
    C: Ord + Clone + Counter,
{
    fn eq(&self, other: &Self) -> bool {
        if let Some(c) = self.partial_cmp(other) {
            matches!(c, Ordering::Equal)
        } else {
            false
        }
    }
}

impl<I, C> FromIterator<(I, C)> for VectorClock<I, C>
where
    I: Hash + Eq + Clone,
    C: Ord + Clone + Counter,
{
    fn from_iter<T: IntoIterator<Item = (I, C)>>(iter: T) -> Self {
        Self {
            map: iter.into_iter().collect(),
        }
    }
}

impl<I, C> VectorClock<I, C>
where
    I: Hash + Eq + Clone,
    C: Ord + Clone + Counter,
{
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn new_incr(id: I) -> Self {
        let mut clock = Self::new();
        clock.increment(&id);
        clock
    }

    /// Increment the clock for the given id. Creates the id if it doesn't exist.
    ///
    /// # Panics
    /// Panics if the clock overflows.
    pub fn increment(&mut self, id: &I) {
        if let Some(counter) = self.map.get_mut(id) {
            counter.increment_expect();
        } else {
            self.map.insert(id.clone(), C::default());
            self.map.get_mut(id).unwrap().increment_expect();
        }
    }

    pub fn get(&self, id: &I) -> Option<&C> {
        self.map.get(id)
    }

    /// Takes two vector clocks and mutates the first such that it contains the
    /// maximum value for each local clock across both vector clocks.
    pub fn sync(&mut self, other: &Self) {
        for (id, counter) in other.map.iter() {
            let v = self.map.entry(id.clone()).or_default();
            *v = max(counter, v).clone();
        }
    }

    /// Takes two vector clocks and produces a new vector clock that contains
    /// the maximum value for each local clock across both vector clocks.
    #[inline]
    pub fn join(first: &Self, second: &Self) -> Self {
        // might be better to use an iterator instead?
        let mut merged = first.clone();
        merged.sync(second);
        merged
    }

    pub fn set_thread_clock(&mut self, thread_id: I, clock: C) {
        self.map.insert(thread_id, clock);
    }
}

impl<I, C> Default for VectorClock<I, C>
where
    I: Hash + Eq + Clone,
    C: Ord + Clone + Counter,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<I, C> PartialOrd for VectorClock<I, C>
where
    I: Hash + Eq + Clone,
    C: Ord + Clone + Counter,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // there's probably a better way to do this but it'll suffice for now
        // not sure if it's better to do extra redundant comparisons or incur
        // the cost of the `unique` call. Something to investigate in the future
        let iter = self.map.keys().chain(other.map.keys()).unique().map(|id| {
            match (self.get(id), other.get(id)) {
                (None, Some(count_other)) => C::default().cmp(count_other),
                (Some(count_self), None) => count_self.cmp(&C::default()),
                (Some(count_self), Some(count_other)) => {
                    count_self.cmp(count_other)
                }
                (None, None) => unreachable!(),
            }
        });

        let mut current_answer = None;
        for cmp in iter {
            if let Some(current_answer) = current_answer.as_mut() {
                match (&current_answer, cmp) {
                    // Incomparable case
                    (Ordering::Less, Ordering::Greater)
                    | (Ordering::Greater, Ordering::Less) => {
                        return None;
                    }
                    (Ordering::Equal, Ordering::Less) => {
                        *current_answer = Ordering::Less;
                    }
                    (Ordering::Equal, Ordering::Greater) => {
                        *current_answer = Ordering::Greater;
                    }
                    _ => {}
                }
            } else {
                current_answer = Some(cmp);
            }
        }

        // If we have an answer, return it. Otherwise, return `Equal` since the
        // `None` case can only happen if the maps are both empty. The
        // incomparable case exits early.
        if let Some(answer) = current_answer {
            Some(answer)
        } else {
            Some(Ordering::Equal)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ValueWithClock {
    pub value: BitVecValue,
    pub clocks: ClockPair,
}

impl ValueWithClock {
    pub fn zero(
        width: u32,
        reading_clock: ClockIdx,
        writing_clock: ClockIdx,
    ) -> Self {
        Self {
            value: BitVecValue::zero(width),
            clocks: ClockPair::new(reading_clock, writing_clock),
        }
    }

    pub fn new(
        value: BitVecValue,
        write_clock: ClockIdx,
        read_clock: ClockIdx,
    ) -> Self {
        Self {
            value,
            clocks: ClockPair::new(read_clock, write_clock),
        }
    }
}

/// A struct containing the read and write clocks for a value. This is small
/// enough to be copied around easily
#[derive(Debug, Clone, PartialEq, Copy)]
pub struct ClockPair {
    pub read_clock: ClockIdx,
    pub write_clock: ClockIdx,
}

impl ClockPair {
    pub fn new(read_clock: ClockIdx, write_clock: ClockIdx) -> Self {
        Self {
            read_clock,
            write_clock,
        }
    }

    pub fn check_read(
        &self,
        (thread, reading_clock): ThreadClockPair,
        clock_map: &mut ClockMap,
    ) -> Result<(), ClockError> {
        if clock_map[reading_clock] >= clock_map[self.write_clock] {
            let v = clock_map[reading_clock][thread];
            clock_map[self.read_clock].set_thread_clock(thread, v);
            Ok(())
        } else if clock_map[reading_clock]
            .partial_cmp(&clock_map[self.write_clock])
            .is_none()
        {
            Err(ClockError::ReadWriteUnhelpful)
        } else {
            // This implies that the read happens before the write which I think
            // shouldn't be possible
            unreachable!(
                "something weird happened. TODO griffin: Sort this out"
            )
        }
    }

    pub fn check_write(
        &self,
        writing_clock: ClockIdx,
        clock_map: &mut ClockMap,
    ) -> Result<(), ClockError> {
        if clock_map[writing_clock] >= clock_map[self.write_clock]
            && clock_map[writing_clock] >= clock_map[self.read_clock]
        {
            clock_map[self.write_clock] = clock_map[writing_clock].clone();
            Ok(())
        } else if clock_map[writing_clock] < clock_map[self.read_clock]
            || clock_map[writing_clock]
                .partial_cmp(&clock_map[self.read_clock])
                .is_none()
        {
            Err(ClockError::ReadWriteUnhelpful)
        } else if clock_map[writing_clock]
            .partial_cmp(&clock_map[self.write_clock])
            .is_none()
        {
            Err(ClockError::WriteWriteUnhelpful)
        } else {
            // This implies the current write happened before the prior write
            // which I think shouldn't be possible
            unreachable!(
                "something weird happened. TODO griffin: Sort this out"
            )
        }
    }
}

#[derive(Debug, Clone, Error)]
pub enum ClockError {
    #[error("Concurrent read & write to the same register/memory")]
    ReadWriteUnhelpful,
    #[error("Concurrent writes to the same register/memory")]
    WriteWriteUnhelpful,
    #[error("Concurrent read & write to the same register/memory {0:?}")]
    ReadWrite(GlobalCellIdx),
    #[error("Concurrent writes to the same register/memory {0:?}")]
    WriteWrite(GlobalCellIdx),
}

impl ClockError {
    pub fn add_cell_info(self, cell: GlobalCellIdx) -> Self {
        match self {
            ClockError::ReadWriteUnhelpful => ClockError::ReadWrite(cell),
            ClockError::WriteWriteUnhelpful => ClockError::WriteWrite(cell),
            _ => self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let clock = VectorClock::<u32>::new_incr(1);
        let mut other = VectorClock::<u32>::new();

        assert!(clock >= other);

        other.increment(&2);

        assert!(clock.partial_cmp(&other).is_none());

        let merged = VectorClock::join(&clock, &other);

        assert!(merged >= clock);
        assert!(merged >= other);
        assert_eq!(VectorClock::<u32>::new(), VectorClock::<u32>::new());
    }

    // Adapted from VClock tests
    #[test]
    fn test_vector_clock_new() {
        let vc1 = VectorClock::<u32>::new_incr(17);
        assert_eq!(None, vc1.get(&0));
        assert_eq!(Some(1), vc1.get(&17).copied());

        let vc2 = VectorClock::<u32>::new_incr(17u32);
        assert_eq!(None, vc2.get(&0u32));
        assert_eq!(Some(1), vc2.get(&17u32).copied());

        let vc3 = VectorClock::<i64, u8>::new_incr(17i64);
        assert_eq!(None, vc3.get(&0i64));
        assert_eq!(Some(1u8), vc3.get(&17i64).copied());
    }

    #[test]
    fn test_vector_clock_increment() {
        let mut vc = VectorClock::<i16>::new();

        assert_eq!(None, vc.get(&0));
        assert_eq!(None, vc.get(&2));

        vc.increment(&0);
        assert_eq!(Some(1), vc.get(&0).copied());
        assert_eq!(None, vc.get(&2));

        vc.increment(&2);
        vc.increment(&0);

        assert_eq!(Some(2), vc.get(&0).copied());
        assert_eq!(Some(1), vc.get(&2).copied());

        vc.increment(&0);
        assert_eq!(Some(3), vc.get(&0).copied());
        assert_eq!(Some(1), vc.get(&2).copied());

        vc.increment(&1);
        assert_eq!(Some(3), vc.get(&0).copied());
        assert_eq!(Some(1), vc.get(&1).copied());
        assert_eq!(Some(1), vc.get(&2).copied());
    }

    #[test]
    fn test_empty_comparison() {
        let vc = VectorClock::<u32>::new();
        let vc2: VectorClock<u32> = [(12, 0), (10, 0)].into_iter().collect();
        let vc3: VectorClock<u32> = [(147, 0), (32, 0)].into_iter().collect();

        assert_eq!(vc, vc2);
        assert_eq!(vc2, vc3);
    }

    #[test]
    #[should_panic]
    fn test_overflow() {
        let mut vc = VectorClock::<u8, u8>::new();
        for _ in 0..257 {
            vc.increment(&0);
        }
    }
}