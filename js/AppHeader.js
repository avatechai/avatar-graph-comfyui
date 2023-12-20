import { van } from "./van.js";
const { div, span } = van.tags;

export function AppHeader() {
  return div(
    {
      class: () => "absolute flex justify-between top-0 w-full text-white p-4",
    },
    div(
      {},
      span(
        {
          class:
            "block bg-gradient-to-b from-gray-500 to-white text-transparent bg-clip-text text-2xl",
        },
        "Avatech v1"
      ),
      span(
        {
          class:
            "bg-gradient-to-b from-gray-500 to-white text-transparent bg-clip-text text-lg",
        },
        "Get your DALLE3 AI Personal Clone"
      )
    ),
    span({ class: "text-gray-300" }, "Twitter")
  );
}
