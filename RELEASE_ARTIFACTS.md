# SpaceForge release artifacts

Every artifact below regenerates from source with the command shown. Nothing
is hand-made. Commit the small ones; rebuild the large ones.

| Artifact | Regenerate | Notes |
|---|---|---|
| `checkpoints/ppo_spacecraft_final.zip` | `python3 -m spaceforge_mvp.training.train --total_timesteps 300000 --save_freq 50000 --seed 1`, then select the snapshot with the best held-out success | Shipped PPO: 100% on `run_eval` seeds; provenance in `ppo_spacecraft_final.metadata.json` |
| `spaceforge_mvp/viz/scenarios/*.json` | `python3 -m spaceforge_mvp.viz.scenarios` | 5 benchmark traces, committed (144 KB) |
| `spaceforge_mvp/viz/dashboard.html` | `python3 -m spaceforge_mvp.viz.dashboard` | Standalone replay, committed |
| `spaceforge_mvp/viz/videos/*.mp4` | `python3 -m spaceforge_mvp.viz.make_videos --film` | NOT committed (~16 MB); 6-9 min render, needs ffmpeg |
| `runs/*/mission.json, telemetry.json, metrics.json` | `python3 -m spaceforge_mvp.simulate --controller ... --output-dir runs/...` | NOT committed (gitignored) |
| `training_report.json` / `hpo_report.json` | rl_trainer / auto_trainer runs | Machine-readable eval/HPO reports |
| `VALIDATION_RESULTS.md` | `python3 -m spaceforge_mvp.eval.validation_matrix --episodes 120 --seed 0` | Measured Phase 6 numbers |
| CI status | `.github/workflows/{ci,frontend,lint,ros}.yml` | `ros.yml` verified in a matching Humble + Harmonic container: 170 tests, 0 failures |

Release checklist: pytest suites green, lint clean, matrix verdicts true,
`ros.yml` green on a ROS runner, dashboard smoke green, artifacts regenerated.
