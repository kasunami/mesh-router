-- Remove obsolete manual state forcing for pupix1/packpup1 lanes.
--
-- pupix1 is an ephemeral dual-boot host and must be represented by explicit
-- operator suspension/expected-offline state, not by forcing stale successful
-- probe rows back to ready.
DROP TRIGGER IF EXISTS trg_mesh_force_pupix_lane_state ON lanes;
DROP FUNCTION IF EXISTS mesh_force_pupix_lane_state();
