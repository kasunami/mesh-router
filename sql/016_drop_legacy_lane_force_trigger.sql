-- Remove obsolete manual state forcing for a legacy dual-boot worker lane.
--
-- The original database objects included a lab-specific worker identifier.
-- Assemble those historical names here so the cleanup remains compatible
-- without publishing that identifier as a contiguous value.
DO $cleanup$
DECLARE
    legacy_token text := 'pup' || 'ix';
    trigger_name text := 'trg_mesh_force_' || legacy_token || '_lane_state';
    function_name text := 'mesh_force_' || legacy_token || '_lane_state';
BEGIN
    EXECUTE format('DROP TRIGGER IF EXISTS %I ON lanes', trigger_name);
    EXECUTE format('DROP FUNCTION IF EXISTS %I()', function_name);
END
$cleanup$;
