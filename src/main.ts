import 'carbon-components-svelte/css/g10.css';
import './app.css';

import { mount } from 'svelte';

import App from './App.svelte';

const target = document.getElementById('app');

if (!target) {
  throw new Error('Application root element was not found');
}

mount(App, { target });
