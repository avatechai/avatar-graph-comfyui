import { van } from "./van.js";
import {
  imageUrl,
  showPreview,
  previewUrl,
  showEditor,
  previewImg,
  previewImgLoading,
} from "./state.js";
const { button, iframe, div, img, input, label, span } = van.tags;
import { app } from "./app.js";
import { uploadPreview } from "./index.js";
import { api } from "./api.js";
// import { uploadSegments } from "./LayerEditor.js";

async function loadJSONWorkflow() {
  const json = await (await fetch("./get_default_workflow")).json();
  app.loadGraphData(json);
  console.log(json);
}

async function uploadImage() {
  /** @type {import('../../../web/types/litegraph.js').LGraph}*/
  const graph = app.graph;
  const nodes = graph.findNodesByType("LoadImage");
  previewImgLoading.val = true;
  console.log(previewImgLoading.val);

  /** @type {any[]}*/
  const widgets = nodes[0].widgets;
  console.log(nodes[0]);
  widgets.find((x) => x.type == "button").callback();
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    if (nodes[0]?.imgs) {
      if (previewImg.val != "" && previewImg.val == nodes[0].imgs[0].currentSrc)
        continue;
      previewImgLoading.val = false;
      return nodes[0].imgs[0].currentSrc;
    }
  }
}

const jsonWorkflowLoading = van.state(true);

async function prepareImageFromUrlRedirect(stage) {
  await new Promise(resolve => setTimeout(resolve, 2000));
  const queue_id = new URLSearchParams(window.location.search).get('queue-id')
  if (
    queue_id && queue_id != ""
  ) {
    console.log(queue_id);
    stage.val = 1
    const graph = app.graph;
    const node = graph.findNodesByType("LoadImage");
    const imageName = "create_avatar_endpoint_" + queue_id + ".png"
    console.log(node[0]);
    node[0].widgets_values[0] = imageName
    node[0].widgets[0].value = imageName
    node[0].widgets[0]._value = imageName
    graph.change()
    previewImg.val = api.apiURL(
      `/view?filename=${encodeURIComponent(
        imageName
      )}&type=input&subfolder=${""}${app.getPreviewFormatParam()}`
    );
    console.log(previewImg);
  }
}

export function AvatarPreview() {
  console.log("getting workflow json now");
  loadJSONWorkflow().then(() => {
    console.log("done loading");
    jsonWorkflowLoading.val = false;
  });

  const loading = van.state(false);
  const shareLoading = van.state(false);

  api.addEventListener("execution_start", (evt) => {
    loading.val = true;
  });

  api.addEventListener("executed", (evt) => {
    loading.val = false;
  });

  const email = van.state("");
  const stage = van.state(0); // 0: upload image, 1: edit segment, 2: generate

  // This will wait 2 seconds until the everything is loaded
  prepareImageFromUrlRedirect(stage)

  const renderSteps = () => {
    return div(
      {
        class: () => "flex flex-col bg-white justify-center",
      },
      div(
        {
          class: () =>
            " bg-gradient-to-b from-black via-[#5F5F5F] via-60% to-white text-transparent bg-clip-text font-gabarito text-4xl",
        },
        "Avatech v1"
      ),
      div(
        {
          class: () =>
            " bg-gradient-to-b from-black via-[#5F5F5F] via-50% to-white text-transparent bg-clip-text font-gabarito text-2xl",
        },
        "Get your DALLE3 AI Personal Clone"
      ),
      // input({
      //   type: "file",
      //   class: "file-input file-input-bordered w-full w-full",
      // }),

      // label(
      //   {
      //     class: () =>
      //       " aspect-square w-full ring-1 ring-black bg-[#f6f6f6] rounded-lg flex justify-center items-center text-black cursor-pointer mt-24",
      //     // for: "imgtosam",
      //     onclick: () => {},
      //   },
      //   div(
      //     {
      //       class: () => " absolute flex justify-center items-center ",
      //     },
      //     "upload your image"
      //   ),
      //   img({
      //     class: () => " z-[10] object-contain w-[550px] h-[394px]",
      //     id: "imgUrl",
      //   })
      // ),
      div(
        {
          class: () =>
            " w-full flex flex-col justify-center items-center gap-4",
        },
        button(
          {
            class: () =>
              "w-full mt-2 btn flex flex-row normal-case px-4 rounded-md left-0 top-0 z-[200] pointer-events-auto ",
            onclick: async () => {
              // previewImg.val = await uploadImage();
              stage.val = 1;
              let input = document.createElement("input");
              input.type = "file";
              input.accept = "image/jpeg,image/png,image/webp";
              document.body.appendChild(input);

              // when the input content changes, do something
              input.onchange = async function (e) {
                if (Object.entries(e.target.files).length) {
                  await uploadFile(e.target.files[0], true);
                }
                previewImg.val = URL.createObjectURL(e.target.files[0]);
                document.body.removeChild(input);
              };
              input.click();
            },
          },
          div({ class: "badge badge-neutral" }, "1"),
          div("Upload your image"),
          span({
            class: "iconify text-lg",
            "data-icon": "material-symbols:drive-folder-upload",
            "data-inline": "false",
          }),
          () =>
            previewImgLoading.val
              ? span({
                  class: "loading loading-spinner loading-md",
                })
              : ""
        ),
        () =>
          previewImg.val != ""
            ? img({
                class: () => "z-[10] object-contain w-[400px] h-[400px] border",
                src: previewImg,
              })
            : "",
        button(
          {
            class: () =>
              "btn w-96 normal-case " + (stage.val < 1 ? "btn-disabled" : ""),
            onclick: () => {
              /** @type {import('../../../web/types/litegraph.js').LGraph}*/
              const graph = app.graph;
              const imageNodes = graph.findNodesByType("LoadImage");
              if (!imageNodes[0].imgs) return;

              const nodes = graph.findNodesByType("SAM MultiLayer");

              /** @type {any[]}*/
              const widgets = nodes[0].widgets;
              console.log(nodes[0]);
              console.log(nodes[0].widgets);
              widgets.find((x) => x.type == "button").callback();
              stage.val = 2;
            },
          },
          div({ class: "badge badge-neutral" }, "2"),
          "Edit segment"
        ),
        button(
          {
            class: () =>
              "btn w-96 normal-case " + (stage.val < 2 ? "btn-disabled" : ""),
            onclick: async () => {
              // const uploaded = await uploadSegments();
              // if (!uploaded) return;

              const graph = app.graph;
              const imageNodes = graph.findNodesByType("LoadImage");
              if (!imageNodes[0].imgs) return;
              document.getElementById("queue-button").click();
            },
          },
          div({ class: "badge badge-neutral" }, "3"),
          () =>
            loading.val
              ? span({
                  class: "loading loading-spinner loading-lg",
                })
              : "Generate"
        )
      )
    );
  };

  const renderIFrame = () => {
    return iframe({
      id: "avatech-viewer-iframe",
      title: "avatech-viewer-iframe",
      name: "avatech-viewer-iframe",
      allow: "cross-origin-isolated",
      class: () =>
        "w-full h-full min-w-[300px] min-h-[600px] z-[100] pointer-events-auto flex border-none overflow-hidden" +
        (showPreview.val ? "" : "hidden"),
      // src: "https://labs.avatech.ai/viewer/default",
      // src: "http://localhost:3000/viewer/default",
      src: previewUrl,
    });
  };

  const renderShareLink = () => {
    return div(
      {
        class: () =>
          "w-full flex flex-col gap-2 justify-center items-center mt-8",
      },
      div(
        {
          class: () =>
            "w-full flex justify-center font-bold italic text-gray-500",
        },
        span("We are launching OpenAI Assistant API integration soon!")
      ),
      div(
        { class: () => "w-[24rem] flex justify-center items-center" },
        input({
          type: "text",
          class: () =>
            "w-full input input-bordered text-black rounded rounded-l-md rounded-r-none !outline-none",
          onchange: (e) => {
            email.val = e.target.value;
          },
          placeholder: "Enter your email",
        }),
        button(
          {
            class: () =>
              "btn rounded rounded-l-none rounded-r-md no-animation bg-neutral hover:bg-neutral-focus text-white border-none normal-case",
            onclick: async () => {
              shareLoading.val = true;
              const url = await (await fetch("./get_webhook")).json();
              await uploadPreview();
              await fetch(url, {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  username: "Avabot",
                  avatar_url:
                    "https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/avatechai.png",
                  content: "New register! \n" + email.val,
                }),
              });
              shareLoading.val = false;
            },
          },
          () =>
            shareLoading.val
              ? span({
                  class: "loading loading-spinner loading-lg",
                })
              : "Get Avatar Link"
        )
      )
    );
  };

  const renderCloseButton = () => {
    return button(
      {
        class: () =>
          "btn flex flex-row btn-ghost text-black normal-case  px-4 rounded-md left-0 top-0 z-[200] w-fit pointer-events-auto ",
        onclick: () => {
          showPreview.val = false;
        },
      },
      span({
        class: "iconify text-lg",
        "data-icon": "ic:round-close",
        "data-inline": "false",
      })
    );
  };

  const renderChangeWorkflowButton = () => {
    return button(
      {
        class: () =>
          "btn text-black flex flex-row btn-ghost normal-case px-4 rounded-md left-0 top-0 z-[200] w-fit pointer-events-auto ",
        onclick: () => {
          let input = document.createElement("input");
          input.type = "file";
          document.body.appendChild(input);
          input.accept = ".json,image/png,.latent,.safetensors";
          input.onchange = async function (e) {
            if (Object.entries(e.target.files).length) {
              app.handleFile(e.target.files[0]);
            }
            document.body.removeChild(input);
          };
          input.click();
          // document.getElementById("comfy-load-button").click();
        },
      },
      span({
        class: "iconify text-lg",
        "data-icon": "ic:round-swap-vert",
        "data-inline": "false",
      }),
      () => (jsonWorkflowLoading.val ? "Loading" : "Change workflow")
    );
  };

  const renderTwitter = () => {
    return button(
      {
        class: () =>
          "absolute top-4 right-4 btn text-black btn-ghost w-24 text-xs !px-0 normal-case",
        onclick: () => window.open("https://twitter.com/avatech_gg", "_blank"),
      },
      "Twitter"
    );
  };

  const isMobileDevice = () => {
    return window.screen.width < 640;
  };

  return div(
    {
      class: () => {
        console.log(showPreview);

        return (
          (showPreview.val ? "" : "hidden ") +
          "w-full h-full absolute left-0 top-0 z-[99] pointer-events-auto flex border-none bg-white"
        );
      },
    },
    div(
      { class: "overflow-y-auto overflow-x-hidden w-full h-full" },
      div(
        {
          class: "absolute top-4 left-4 flex flex-row gap-2",
        },
        renderCloseButton(),
        renderChangeWorkflowButton()
      ),
      renderTwitter(),
      () => {
        if (isMobileDevice()) {
          return div(
            {
              class: () =>
                "flex flex-col w-full h-fit bg-white justify-center items-center py-16 px-4 gap-2" +
                (showPreview.val ? "" : "hidden"),
            },
            renderIFrame(),
            renderSteps(),
            renderShareLink()
          );
        } else {
          return div(
            {
              class: () =>
                "flex w-full h-full bg-white justify-around items-center p-24" +
                (showPreview.val ? "" : "hidden"),
            },
            renderSteps(),
            div(
              { class: () => "flex flex-col" },
              renderIFrame(),
              renderShareLink()
            )
          );
        }
      }
    )
  );
}

function showImage(name) {
  const graph = app.graph;
  const node = graph.findNodesByType("LoadImage");
  const img = new Image();
  img.onload = () => {
    node[0].imgs = [img];
    app.graph.setDirtyCanvas(true);
  };
  let folder_separator = name.lastIndexOf("/");
  let subfolder = "";
  if (folder_separator > -1) {
    subfolder = name.substring(0, folder_separator);
    name = name.substring(folder_separator + 1);
  }
  img.src = api.apiURL(
    `/view?filename=${encodeURIComponent(
      name
    )}&type=input&subfolder=${subfolder}${app.getPreviewFormatParam()}`
  );
  node.setSizeForImage?.();
}

async function uploadFile(file, updateNode, pasted = false) {
  try {
    // Wrap file in formdata so it includes filename
    const graph = app.graph;
    const nodes = graph.findNodesByType("LoadImage");
    const widgets = nodes[0].widgets.find((w) => w.name === "image");
    const body = new FormData();
    body.append("image", file);
    if (pasted) body.append("subfolder", "pasted");
    const resp = await api.fetchApi("/upload/image", {
      method: "POST",
      body,
    });

    if (resp.status === 200) {
      const data = await resp.json();
      // Add the file to the dropdown list and update the widget value
      let path = data.name;
      if (data.subfolder) path = data.subfolder + "/" + path;

      if (!widgets.options.values.includes(path)) {
        widgets.options.values.push(path);
      }

      if (updateNode) {
        showImage(path);
        widgets.value = path;
      }
    } else {
      alert(resp.status + " - " + resp.statusText);
    }
  } catch (error) {
    alert(error);
  }
}
