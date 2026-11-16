"""Read-only checkpoint comparison in CPU MuJoCo, generated for the 2026-11-16 audit."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import time
import torch
from supermimic.wf.evaluate import config_from_dict, evaluate_policy
from supermimic.wf.goal_trajectory import TrajectoryLibrary
from supermimic.wf.networks import Actor
from supermimic.wf.robot import build_model
from supermimic.wf.task import WholeBodyRewardCfg, WholeBodyTask
from supermimic.wf.train import model_manifest

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = {
 'old': ROOT/'outputs/wf_cmd_gpu_20261115_133130_363428',
 'new': ROOT/'outputs/wf_cmd_gpu_20261115_220027_884431',
}

def main():
 torch.set_num_threads(4)
 checkpoints = {}
 for label, directory in RUNS.items():
  path = directory/'ckpt_latest.pt'
  ck = torch.load(path,map_location='cpu',weights_only=False)
  cfg = config_from_dict(ck['config'])
  library_path=ROOT/cfg.trajectory.library
  library_manifest=json.loads((library_path/'manifest.json').read_text())
  checkpoints[label]=(path,ck,cfg,library_path,library_manifest)
 assert len({x[1]['manifest']['physics_hash'] for x in checkpoints.values()})==1
 assert len({x[4]['sha256'] for x in checkpoints.values()})==1
 assert checkpoints['old'][1]['reward_config']==checkpoints['new'][1]['reward_config']
 summary = {}
 for label,(path,ck,cfg,library_path,library_manifest) in checkpoints.items():
  model,spec=build_model(cfg.asset_dir,cfg.sim.timestep,
      leg_action_scale=cfg.sim.leg_action_scale,
      arm_action_scale=cfg.sim.arm_action_scale,
      wheel_action_scale=cfg.sim.wheel_action_scale)
  assert model_manifest(spec)['physics_hash']==ck['manifest']['physics_hash']
  task=WholeBodyTask(model,spec,'cpu',WholeBodyRewardCfg(**ck['reward_config']))
  actor=Actor(cfg.policy).to('cpu')
  actor.load_state_dict(ck['actor'])
  library=TrajectoryLibrary(library_path,task,ck['manifest']['physics_hash'],cfg.sim.dt,'cpu')
  for level in (1,3):
   start=time.monotonic()
   print(json.dumps(dict(event='start',label=label,level=level,iteration=ck['iteration'])),flush=True)
   result=evaluate_policy(actor,task,model,spec,episodes=16,level=level,seed=10000,
      steps=600,hold_steps=cfg.curriculum.hold_steps,dt=cfg.sim.dt,friction=1.,
      settle_steps=cfg.sim.reset_settle_steps,goal_seed_device='cpu',
      arm_initial_position=cfg.sim.arm_initial_position,
      trajectories=library,trajectory_cfg=cfg.trajectory)
   result.update(checkpoint=str(path),checkpoint_iteration=ck['iteration'],
      physics_hash=ck['manifest']['physics_hash'],trajectory_npz_sha256=library_manifest['sha256'],
      actor_sha256=hashlib.sha256(b''.join(t.cpu().numpy().tobytes() for t in ck['actor'].values())).hexdigest(),
      checkpoint_training_metadata={k:ck.get(k) for k in ('actor_updates','accepted_actor_updates','pending_actor_updates','level','guard_level','guard_verified_iteration')},
      evaluation_wall_seconds=time.monotonic()-start,
      protocol='CPU MuJoCo; 16 episodes; seed10000; steps600; friction1; all trajectory schedule with same NPZ and reward configuration; own original checkpoint actor; latest file snapshotted in memory at evaluation start.')
   output=OUT/f'paired_eval_{label}_level{level}.json'
   output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
   compact={k:result[k] for k in ('checkpoint_iteration','tracking_success_rate','tracking_rms_position_error_m','tracking_rms_orientation_error_rad','failure_rate','held_success_rate','final_held_success_rate','target_hash','evaluation_wall_seconds')}
   summary[f'{label}_level{level}']=compact
   print(json.dumps(dict(event='complete',label=label,level=level,**compact)),flush=True)
 for level in (1,3):
  assert summary[f'old_level{level}']['target_hash']==summary[f'new_level{level}']['target_hash']
 (OUT/'paired_eval_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 print('All comparisons complete. Initial target hashes match within each difficulty level.',flush=True)

if __name__=='__main__':main()
