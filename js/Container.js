import { ImageEditor } from './ImageEditor.js';
import { AvatarEditor } from './AvatarEditor.js';
import { van } from './van.js';
const { button, iframe, div, img } = van.tags;

export function Container() {
  return div(
    {
      class: 'fixed left-0 top-0 w-full h-full z-[200] pointer-events-none',
      id: 'avatech-editor',
    },
    AvatarEditor(),
    ImageEditor()
  );
}
