import type { Action } from "@elizaos/core";
import { bowAction } from "./bow";
import { loadModeAction, unloadModeAction } from "./loadMode";
import { runActionGroupAction } from "./runActionGroup";
import { runSkillAction } from "./runSkill";
import { setHeadAction } from "./setHead";
import { setServoAction } from "./setServo";
import { sideStepLeftAction } from "./sideStepLeft";
import { sideStepRightAction } from "./sideStepRight";
import { squatAction } from "./squat";
import { standAction } from "./stand";
import { standUpAction } from "./standUp";
import { stopAction } from "./stop";
import { turnLeftAction } from "./turnLeft";
import { turnRightAction } from "./turnRight";
import { walkBackwardAction } from "./walkBackward";
import { walkForwardAction } from "./walkForward";
import { waveAction } from "./wave";

export {
  bowAction,
  loadModeAction,
  runActionGroupAction,
  runSkillAction,
  setHeadAction,
  setServoAction,
  sideStepLeftAction,
  sideStepRightAction,
  squatAction,
  standAction,
  standUpAction,
  stopAction,
  turnLeftAction,
  turnRightAction,
  unloadModeAction,
  walkBackwardAction,
  walkForwardAction,
  waveAction,
};

export const actions: Action[] = [
  walkForwardAction,
  walkBackwardAction,
  sideStepLeftAction,
  sideStepRightAction,
  turnLeftAction,
  turnRightAction,
  stopAction,
  standAction,
  standUpAction,
  squatAction,
  waveAction,
  bowAction,
  setHeadAction,
  setServoAction,
  runActionGroupAction,
  loadModeAction,
  unloadModeAction,
  runSkillAction,
];
