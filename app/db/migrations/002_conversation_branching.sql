-- SPDX-License-Identifier: Apache-2.0
-- Add parent conversation reference and forked message ID for conversation branching / time travel
ALTER TABLE conversations ADD COLUMN parent_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL;
ALTER TABLE conversations ADD COLUMN forked_from_message_id TEXT;
CREATE INDEX IF NOT EXISTS idx_conversations_parent ON conversations(parent_conversation_id);
