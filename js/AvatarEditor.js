import { iframeSrc, showEditor } from './state.js';
import { van } from './van.js';
const { button, iframe, div, img } = van.tags;

export function AvatarEditor() {
  return div(
    {
      class: 'w-full h-full',
    },
    button(
      {
        class: () =>
          'px-4 py-2 rounded-md absolute left-0 top-0 z-[100] pointer-events-auto ' +
          (showEditor.val ? '' : 'hidden'),
        onclick: () => {
          console.log('close');
          showEditor.val = false;
        },
      },
      'back',
    ),
    iframe({
      id: 'avatech-editor-iframe',
      title: 'avatech-editor-iframe',
      name: 'avatech-editor-iframe',
      class: () =>
        'w-full h-full pointer-events-auto ' + (showEditor.val ? '' : 'hidden'),
      src: iframeSrc,
    }),
  );
}
