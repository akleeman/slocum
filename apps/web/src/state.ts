export type SelectionState = {
  cycle: string;
  leadHour: number;
};

export type SelectionAction =
  | { type: "setCycle"; cycle: string }
  | { type: "setLead"; leadHour: number };

export function selectionReducer(
  state: SelectionState,
  action: SelectionAction
): SelectionState {
  if (action.type === "setCycle") {
    return { ...state, cycle: action.cycle };
  }
  if (action.type === "setLead") {
    return { ...state, leadHour: action.leadHour };
  }
  return state;
}

export function tileRequestKey(state: SelectionState, z: number, x: number, y: number): string {
  return `${state.cycle}:${state.leadHour}:${z}:${x}:${y}`;
}
