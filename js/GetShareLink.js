import { van } from "./van.js";
const { button, div, span, input } = van.tags;

export function GetShareLink() {
  return div(
    {
      class: () =>
        "absolute flex flex-col justify-center items-center top-0 left-0 bg-gray-900 bg-opacity-50 pointer-events-auto w-full h-full gap-2",
    },
    span("We're launching OpenAI Assistant API integration soon!"),
    div(
      {
        class: "w-[24rem] flex justify-center items-center",
      },
      input({
        class:
          "w-full input input-bordered text-black rounded rounded-l-md rounded-r-none",
        placeholder: "Email",
      }),
      button(
        {
          class:
            "btn rounded rounded-l-none rounded-r-md no-animation bg-neutral hover:bg-neutral-focus text-white border-none normal-case",
        },
        "Get Avatar Link"
      )
    )
  );
}
