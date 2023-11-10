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

  return div(
    {
      class: () => {
        console.log(showPreview);

        return (
          (showPreview.val ? "" : "hidden ") +
          "w-full h-screen absolute left-0 top-0 z-[99] pointer-events-auto flex border-none "
        );
      },
    },
    div(
      {
        class: "absolute top-4 left-4 flex flex-row gap-2",
      },
      button(
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
      ),
      button(
        {
          class: () =>
            "btn text-black flex flex-row btn-ghost normal-case  px-4 rounded-md left-0 top-0 z-[200] w-fit pointer-events-auto ",
          onclick: () => {
            document.getElementById("comfy-load-button").click();
          },
        },
        span({
          class: "iconify text-lg",
          "data-icon": "ic:round-swap-vert",
          "data-inline": "false",
        }),
        () => (jsonWorkflowLoading.val ? "Loading" : "Change workflow")
      )
    ),
    button(
      {
        class: () =>
          "absolute top-4 right-4 btn text-black btn-ghost w-24 text-xs !px-0 normal-case",
        onclick: () => window.open("https://twitter.com/avatech_gg", "_blank"),
      },
      "Twitter"
    ),
    div(
      {
        class: () =>
          "flex w-full h-full bg-white justify-around items-center p-24" +
          (showPreview.val ? "" : "hidden"),
      },
      div(
        {
          class: () => "flex flex-col bg-white justify-center h-full",
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
                previewImg.val = await uploadImage();
              },
            },
            div({ class: "badge badge-neutral" }, "1"),
            div("Upload your image"),
            span({
              class: "iconify text-lg",
              "data-icon": "material-symbols:drive-folder-upload",
              "data-inline": "false",
            }),
            () => (previewImgLoading.val
                ? span({
                    class: "loading loading-spinner loading-lg",
                  })
                : '')
          ),
          () =>
            previewImg.val != ""
              ? img({
                  class: () => "z-[10] object-contain w-full h-[394px] border",
                  src: previewImg,
                })
              : "",
          button(
            {
              class: () => "btn w-96 normal-case",
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
              },
            },
            div({ class: "badge badge-neutral" }, "2"),
            "Edit segment"
          ),
          button(
            {
              class: () => "btn w-96 normal-case",
              onclick: () => {
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
      ),
      div(
        { class: () => "flex flex-col" },
        iframe({
          id: "avatech-viewer-iframe",
          title: "avatech-viewer-iframe",
          name: "avatech-viewer-iframe",
          allow: "cross-origin-isolated",
          class: () =>
            "w-[450px] h-[450px] z-[100] pointer-events-auto flex  border-none overflow-hidden" +
            (showPreview.val ? "" : "hidden"),
          // src: "https://labs.avatech.ai/viewer/default",
          // src: "http://localhost:3000/viewer/default",
          src: previewUrl,
        }),
        div(
          { class: () => "w-full flex flex-col gap-2 justify-center mt-24" },
          div(
            { class: () => "flex text-black font-bold" },
            div({}, "We are launching OpenAI Assistant API integration soon")
          ),
          div(
            { class: () => "flex justify-center" },
            input({
              type: "text",
              class: () =>
                "border !border-r-0 border-black rounded-l-lg ring-black/30 p-1 text-black h-12   ",
              onchange: (e) => {
                email.val = e.target.value;
              },
              placeholder: "Enter your email",
            }),
            button(
              {
                class: () =>
                  "btn btn-outline !rounded-l-none !rounded-r-lg p-1 normal-case",
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
        )
      )
    )
  );
}
