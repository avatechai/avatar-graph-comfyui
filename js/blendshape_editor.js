import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';
import van from 'https://cdn.jsdelivr.net/gh/vanjs-org/van/public/van-1.1.3.min.js';

function addMenuHandler(nodeType, cb) {
  const getOpts = nodeType.prototype.getExtraMenuOptions;
  nodeType.prototype.getExtraMenuOptions = function () {
    const r = getOpts.apply(this, arguments);
    cb.apply(this, arguments);
    return r;
  };
}

const iframeSrc = van.state('https://editor.avatech.ai');
const showEditor = van.state(false);

function openInAvatechEditor(url, fileName) {
  let editor = document.getElementById('avatech-editor-iframe');
  iframeSrc.val = url;
  showEditor.val = true;

  editor.contentWindow.postMessage(
    {
      key: 'key',
      value: fileName,
      method: 'store',
    },
    '*',
  );
}

app.registerExtension({
  name: 'Avatech.Avatar.BlendshapeEditor',
  init(app) {},

  async setup() {
    const { button, iframe, div } = van.tags;

    const AvatarEditor = () => {
      return div(
        {
          class: 'fixed left-0 top-0 w-full h-full z-[200] pointer-events-none',
          id: 'avatech-editor',
        },
        button(
          {
            class: () =>
              'px-4 py-2 rounded-md absolute left-0 top-0 z-[100] pointer-events-auto ' + (showEditor.val ? '' : 'hidden'),
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
            'w-full h-full pointer-events-auto ' +
            (showEditor.val ? '' : 'hidden'),
          src: iframeSrc,
        }),
      );
    };

    van.add(document.body, AvatarEditor());
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === 'ExportGLTF') {
      addMenuHandler(nodeType, function (_, options) {
        const output = app.nodeOutputs[this.id + ""];
        if (!output || !output.gltfFilename) return;

        const gltfFilename = window.location.protocol + '//' + api.api_host + api.api_base + (`/view?filename=${output.gltfFilename[0]}`) 

        options.unshift({
          content: 'Save file',
          callback: () => {
            const a = document.createElement('a');
            let url = new URL(gltfFilename);
            url.searchParams.delete('preview');
            a.href = url;
            a.setAttribute(
              'download',
              new URLSearchParams(url.search).get('filename'),
            );
            document.body.append(a);
            a.click();
            requestAnimationFrame(() => a.remove());
          },
        });

        options.unshift({
          content: 'Open In Avatech Editor (Local)',
          callback: () => {
            openInAvatechEditor('http://localhost:3006', gltfFilename);
          },
        });

        options.unshift({
          content: 'Open In Avatech Editor',
          callback: () => {
            openInAvatechEditor('https://editor.avatech.ai', gltfFilename);
          },
        });
      });
    }
  },
});
