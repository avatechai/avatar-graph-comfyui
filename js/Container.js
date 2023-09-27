import { ImageEditor } from './ImageEditor.js';
import { AvatarEditor } from './AvatarEditor.js';
import { van } from './van.js';
import { AvatarViewer } from './AvatarViewer.js';
import { Loading } from './Loading.js';
import { Alert } from './Alert.js';
const { button, iframe, div, img } = van.tags;

export function Container() {
  return div(
    {
      class: 'fixed left-0 top-0 w-full h-full z-[200] pointer-events-none',
      id: 'avatech-editor',
    },
    AvatarEditor(),
    ImageEditor(),
    AvatarViewer(),
    Loading(),
    Alert(),
  );
}
