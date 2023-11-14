import { alertDialog } from "./state.js";
import { van } from "./van.js";
const { div, span } = van.tags;

export function Alert() {
  van.derive(() => {
    if (alertDialog.val.time > 0) {
      setTimeout(() => {
        alertDialog.val = { text: "", time: 0 };
      }, alertDialog.val.time);
    }
  });

  return div(
    {
      class: () =>
        "absolute z-[100] bottom-8 flex justify-center w-full " +
        (alertDialog.val.text ? "" : "hidden"),
    },
    div(
      {
        class:
          "bg-orange-100 border-t-4 border-orange-500 rounded-sm text-orange-700 p-2",
      },
      () => span(alertDialog.val.text)
    )
  );
}
