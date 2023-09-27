import { loadingCaption, showLoading } from "./state.js";
import { van } from "./van.js";
const { div, span } = van.tags;

export function Loading() {
  return div(
    {
      class: () =>
        "absolute flex flex-col justify-center items-center top-0 left-0 bg-gray-900 bg-opacity-50 pointer-events-auto w-full h-full " +
        (showLoading.val ? "" : "hidden"),
    },
    span({
      class: "loading loading-spinner loading-lg mb-2",
    }),
    () => span(loadingCaption.val)
  );
}
