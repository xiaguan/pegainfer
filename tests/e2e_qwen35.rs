// Qwen3.5 e2e tests are deferred until paged KV migration is complete
// and the scheduler supports Qwen3.5.

#[test]
#[ignore = "Qwen3.5 scheduler not yet implemented"]
fn test_e2e_qwen35_generation() {
    // TODO: re-enable when Qwen3.5 is migrated to paged KV + scheduler
}
