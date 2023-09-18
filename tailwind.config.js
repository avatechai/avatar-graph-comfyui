/** @type {import('tailwindcss').Config} */
module.exports = {
  // content: ['./js/**/*.{html,js}'],
  content: ['./js/**/*.{html,js}'],
  theme: {
    extend: {},
  },
  daisyui: {
    themes: ['light'],
  },
  plugins: [require('daisyui')],
};
