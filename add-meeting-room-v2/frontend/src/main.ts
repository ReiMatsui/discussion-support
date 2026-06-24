import { mountExperimentalViewer } from "./experimental-viewer/viewer";
import "./styles.css";

const root = document.querySelector<HTMLElement>("#app");

if (root === null) {
  throw new Error("app root is missing");
}

mountExperimentalViewer(root);
