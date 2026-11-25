//! Explicit ActionState typestate machine and per-region work state.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionState {
    Candidate,
    Ready,
    Submitted(u64),
    Completed,
    Invalidated,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionError {
    IllegalTransition {
        from: ActionState,
        to: &'static str,
    },
}

impl ActionState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            ActionState::Completed | ActionState::Invalidated | ActionState::Rejected
        )
    }

    pub fn mark_ready(self) -> Result<Self, TransitionError> {
        match self {
            ActionState::Candidate | ActionState::Ready => Ok(ActionState::Ready),
            other => Err(TransitionError::IllegalTransition {
                from: other,
                to: "Ready",
            }),
        }
    }

    pub fn submit(self, ticket: u64) -> Result<Self, TransitionError> {
        match self {
            ActionState::Ready => Ok(ActionState::Submitted(ticket)),
            other => Err(TransitionError::IllegalTransition {
                from: other,
                to: "Submitted",
            }),
        }
    }

    pub fn complete(self) -> Result<Self, TransitionError> {
        match self {
            ActionState::Submitted(_) => Ok(ActionState::Completed),
            other => Err(TransitionError::IllegalTransition {
                from: other,
                to: "Completed",
            }),
        }
    }

    pub fn invalidate(self) -> Result<Self, TransitionError> {
        match self {
            ActionState::Candidate | ActionState::Ready | ActionState::Submitted(_) => {
                Ok(ActionState::Invalidated)
            }
            other => Err(TransitionError::IllegalTransition {
                from: other,
                to: "Invalidated",
            }),
        }
    }

    pub fn reject(self) -> Result<Self, TransitionError> {
        match self {
            ActionState::Candidate | ActionState::Ready | ActionState::Submitted(_) => {
                Ok(ActionState::Rejected)
            }
            other => Err(TransitionError::IllegalTransition {
                from: other,
                to: "Rejected",
            }),
        }
    }
}

pub const DIRTY_NONE: u32 = 0;
pub const DIRTY_EVIDENCE: u32 = 1 << 0;
pub const DIRTY_POSE: u32 = 1 << 1;
pub const DIRTY_CONTRADICTION: u32 = 1 << 2;
pub const DIRTY_STRUCTURAL: u32 = 1 << 3;

#[derive(Debug, Clone, Default)]
pub struct RegionState {
    pub map_version: u64,
    pub pose_version: u64,
    pub evidence_version: u64,
    pub dirty_flags: u32,
    pub pending_gain: f32,
    pub inflight_ops: u32,
    pub completed_ops: u32,
    pub frozen: bool,
    pub retracted: bool,
    pub exclusive_locked: bool,
    pub last_evidence_ingested_ms: f64,
}
