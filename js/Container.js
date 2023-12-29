import { LayerEditor } from "./LayerEditor.js";
import { ShapeFlowEditor } from "./ShapeFlowEditor.js";
import { van } from "./van.js";
import { AvatarPreview } from "./AvatarPreview.js";
import { Loading } from "./Loading.js";
import { Alert } from "./Alert.js";
import { AppHeader } from "./AppHeader.js";
import { CombinePointsDialog } from "./CombinePointsDialog.js";
const { button, iframe, div, img } = van.tags;

export function Container() {
  return div(
    {
      class: "fixed left-0 top-0 w-full h-full z-[1000] pointer-events-none",
      id: "avatech-editor",
    },
    CombinePointsDialog(),
    ShapeFlowEditor(),
    LayerEditor(),
    AvatarPreview(),
    Loading(),
    Alert()
  );
}
