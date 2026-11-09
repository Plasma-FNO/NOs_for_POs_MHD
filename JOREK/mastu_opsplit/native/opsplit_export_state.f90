! Diagnostic-only model303 export. This does not implement the native backend.
integer, save :: opsplit_export_element_id = 0
integer, save :: opsplit_export_unit = 0
logical, save :: opsplit_export_initialized = .false.
logical, save :: opsplit_export_enabled = .false.
logical, allocatable, save :: opsplit_export_seen(:)
