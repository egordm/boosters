//! Conversion utilities between training-time and inference-time representations.
//!
//! **Phase 2 Update**: This module is now largely obsolete as tree grower directly builds
//! inference-ready `TreeStorage` during training. The conversion step has been eliminated
//! for improved performance and simpler architecture.
//!
//! This file is kept for reference and potential future use cases where conversion
//! between internal representations might be needed.
