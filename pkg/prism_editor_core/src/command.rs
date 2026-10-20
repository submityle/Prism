use alloc::{string::String, vec::Vec};
use core::fmt;

/// Sorted, duplicate-free domain names affected by an editor command.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AffectSet {
    domains: Vec<String>,
}

impl AffectSet {
    /// Creates an empty set of affected domains.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a domain name and keeps the set sorted.
    pub fn insert(&mut self, domain: impl Into<String>) {
        let domain = domain.into();
        if !self.domains.iter().any(|entry| entry == &domain) {
            self.domains.push(domain);
            self.domains.sort();
        }
    }

    /// Returns the affected domain names in sorted order.
    #[must_use]
    pub fn domains(&self) -> &[String] {
        &self.domains
    }

    /// Returns whether this set shares at least one domain with `other`.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.domains
            .iter()
            .any(|domain| other.domains.binary_search(domain).is_ok())
    }
}

/// Marker trait for the mutable state edited by commands.
pub trait EditContext {}

/// User-facing failure returned by a command operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandError {
    message: String,
}

impl CommandError {
    /// Creates an error with the supplied message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl core::error::Error for CommandError {}

/// Result of asking a command to merge a following command.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MergeResult {
    /// The following command was incorporated into the previous command.
    Merged,
    /// The commands must remain separate history entries.
    NotMergeable,
}

/// Result reported after a command is applied to a context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandStatus {
    /// The command was appended as a new history entry.
    Applied,
    /// The command was merged into the previous history entry.
    Merged,
}

/// Reversible operation over an [`EditContext`].
pub trait Command<C: EditContext>: Send {
    /// Applies the command to `context`.
    fn apply(&mut self, context: &mut C) -> Result<(), CommandError>;
    /// Reverses the command in `context`.
    fn undo(&mut self, context: &mut C) -> Result<(), CommandError>;
    /// Returns the stable label shown in editor history.
    fn label(&self) -> &str;
    /// Returns the domains invalidated by this command.
    fn affected(&self) -> &AffectSet;
    /// Attempts to merge `next` into this command.
    fn merge_from(&mut self, next: &dyn Command<C>) -> MergeResult;
}

struct TransactionCommand<C: EditContext> {
    label: String,
    commands: Vec<Box<dyn Command<C>>>,
    affected: AffectSet,
}

impl<C: EditContext> TransactionCommand<C> {
    fn new(label: impl Into<String>, commands: Vec<Box<dyn Command<C>>>) -> Self {
        let mut affected = AffectSet::new();
        for command in &commands {
            for domain in command.affected().domains() {
                affected.insert(domain.clone());
            }
        }
        Self {
            label: label.into(),
            commands,
            affected,
        }
    }
}

impl<C: EditContext> Command<C> for TransactionCommand<C> {
    fn apply(&mut self, context: &mut C) -> Result<(), CommandError> {
        let mut applied = 0;
        while applied < self.commands.len() {
            if let Err(error) = self.commands[applied].apply(context) {
                for command in self.commands[..applied].iter_mut().rev() {
                    let _ = command.undo(context);
                }
                return Err(error);
            }
            applied += 1;
        }
        Ok(())
    }

    fn undo(&mut self, context: &mut C) -> Result<(), CommandError> {
        for command in self.commands.iter_mut().rev() {
            command.undo(context)?;
        }
        Ok(())
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn affected(&self) -> &AffectSet {
        &self.affected
    }

    fn merge_from(&mut self, _next: &dyn Command<C>) -> MergeResult {
        MergeResult::NotMergeable
    }
}

/// A group of commands applied and undone as one history entry.
pub struct Transaction<C: EditContext + 'static> {
    label: String,
    commands: Vec<Box<dyn Command<C>>>,
}

impl<C: EditContext + 'static> Transaction<C> {
    /// Creates an empty transaction with a history label.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            commands: Vec::new(),
        }
    }

    /// Appends a command to the transaction.
    pub fn push(&mut self, command: impl Command<C> + 'static) {
        self.commands.push(Box::new(command));
    }

    /// Returns whether the transaction contains no commands.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    fn finish(self) -> Option<TransactionCommand<C>> {
        (!self.commands.is_empty()).then(|| TransactionCommand::new(self.label, self.commands))
    }
}

/// Undo/redo history for commands over one editor context.
pub struct CommandStack<C: EditContext + 'static> {
    undo: Vec<Box<dyn Command<C>>>,
    redo: Vec<Box<dyn Command<C>>>,
}

impl<C: EditContext + 'static> Default for CommandStack<C> {
    fn default() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }
}

impl<C: EditContext + 'static> CommandStack<C> {
    /// Applies a command and records it, merging with the previous entry when supported.
    pub fn execute(
        &mut self,
        mut command: Box<dyn Command<C>>,
        context: &mut C,
    ) -> Result<CommandStatus, CommandError> {
        command.apply(context)?;
        let status = if let Some(previous) = self.undo.last_mut() {
            match previous.merge_from(&*command) {
                MergeResult::Merged => CommandStatus::Merged,
                MergeResult::NotMergeable => {
                    self.undo.push(command);
                    CommandStatus::Applied
                }
            }
        } else {
            self.undo.push(command);
            CommandStatus::Applied
        };
        self.redo.clear();
        Ok(status)
    }

    /// Applies all commands in a transaction and records them as one entry.
    pub fn execute_transaction(
        &mut self,
        transaction: Transaction<C>,
        context: &mut C,
    ) -> Result<CommandStatus, CommandError> {
        let Some(command) = transaction.finish() else {
            return Ok(CommandStatus::Applied);
        };
        self.execute(Box::new(command), context)
    }

    /// Undoes the most recent history entry, returning whether one existed.
    pub fn undo(&mut self, context: &mut C) -> Result<bool, CommandError> {
        let Some(mut command) = self.undo.pop() else {
            return Ok(false);
        };
        if let Err(error) = command.undo(context) {
            self.undo.push(command);
            return Err(error);
        }
        self.redo.push(command);
        Ok(true)
    }

    /// Reapplies the most recently undone history entry, returning whether one existed.
    pub fn redo(&mut self, context: &mut C) -> Result<bool, CommandError> {
        let Some(mut command) = self.redo.pop() else {
            return Ok(false);
        };
        if let Err(error) = command.apply(context) {
            self.redo.push(command);
            return Err(error);
        }
        self.undo.push(command);
        Ok(true)
    }

    /// Returns the number of entries currently available to undo.
    #[must_use]
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    /// Returns the number of entries currently available to redo.
    #[must_use]
    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Context {
        value: i32,
    }
    impl EditContext for Context {}

    struct Add {
        amount: i32,
        affect: AffectSet,
    }

    impl Add {
        fn new(amount: i32) -> Self {
            let mut affect = AffectSet::new();
            affect.insert("document");
            Self { amount, affect }
        }
    }

    impl Command<Context> for Add {
        fn apply(&mut self, context: &mut Context) -> Result<(), CommandError> {
            context.value += self.amount;
            Ok(())
        }
        fn undo(&mut self, context: &mut Context) -> Result<(), CommandError> {
            context.value -= self.amount;
            Ok(())
        }
        fn label(&self) -> &str {
            "add"
        }
        fn affected(&self) -> &AffectSet {
            &self.affect
        }
        fn merge_from(&mut self, _next: &dyn Command<Context>) -> MergeResult {
            MergeResult::NotMergeable
        }
    }

    #[test]
    fn stack_round_trips_transaction_and_redo() {
        let mut stack = CommandStack::default();
        let mut context = Context::default();
        let mut transaction = Transaction::new("pair");
        transaction.push(Add::new(2));
        transaction.push(Add::new(3));
        stack
            .execute_transaction(transaction, &mut context)
            .expect("apply");
        assert_eq!(context.value, 5);
        assert_eq!(stack.undo_len(), 1);
        stack.undo(&mut context).expect("undo");
        assert_eq!(context.value, 0);
        stack.redo(&mut context).expect("redo");
        assert_eq!(context.value, 5);
    }
}
