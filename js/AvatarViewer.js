import { iframeSrc, showViewer } from './state.js'
import { van } from './van.js'
const { button, iframe, div, img } = van.tags

export function AvatarViewer() {
  return div(
    {
      class: 'w-full h-full max-w-96 max-h-96 absolute right-0 top-0 z-[100] pointer-events-auto',
    },
    iframe({
      id: 'avatech-viewer-iframe',
      title: 'avatech-viewer-iframe',
      name: 'avatech-viewer-iframe',
      class: () => 'w-full h-full flex pointer-events-auto ',
      src: 'https://labs.avatech.ai/viewer/default',
      // src: 'http://localhost:3000/viewer/default',
    })
  )
}
