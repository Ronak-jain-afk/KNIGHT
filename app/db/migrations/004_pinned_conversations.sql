-- Add pinned column to conversations table for anchoring favorite chats to top of sidebar
ALTER TABLE conversations ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_conversations_pinned ON conversations(pinned DESC, updated_at DESC);
