import { alertDialog } from "./state.js";
import { van } from "./van.js";
const { div, span } = van.tags;

export function Alert() {
  const color = van.state("bg-orange-100 text-orange-700 border-orange-500");

  van.derive(() => {
    if (alertDialog.val.time > 0) {
      switch (alertDialog.val.type) {
        case "error":
          color.val = "bg-red-100 text-red-700 border-red-500";
          break;
        case "success":
          color.val = "bg-green-100 text-green-700 border-green-500";
          break;
        case "info":
          color.val = "bg-blue-100 text-blue-700 border-blue-500";
          break;
        case "warning":
        default:
          color.val = "bg-orange-100 text-orange-700 border-orange-500";
          break;
      }
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
        class: () => `${color.val} border-t-4 rounded-sm p-2`,
      },
      () => span(alertDialog.val.text)
    )
  );
}
