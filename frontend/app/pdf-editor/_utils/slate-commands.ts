/**
 * High-level Slate editor commands used by the toolbar and page.
 *
 * Replaces all document.execCommand() calls.
 */

import { Editor, Transforms, Element as SlateElement } from 'slate';
import type { CustomEditor, BlockType, AlignType, CustomText } from './slate-types';

const LIST_TYPES: BlockType[] = ['bulleted-list', 'numbered-list'];

// ─────────────────────────────────────────────────────────────────────────────
// Mark helpers
// ─────────────────────────────────────────────────────────────────────────────

type TogglableMark = 'bold' | 'italic' | 'underline' | 'strikethrough';

export function isMarkActive(editor: CustomEditor, mark: TogglableMark): boolean {
  const marks = Editor.marks(editor);
  return marks ? marks[mark] === true : false;
}

export function toggleMark(editor: CustomEditor, mark: TogglableMark): void {
  if (isMarkActive(editor, mark)) {
    Editor.removeMark(editor, mark);
  } else {
    Editor.addMark(editor, mark, true);
  }
}

export function setMark(editor: CustomEditor, key: keyof Omit<CustomText, 'text'>, value: string | true | undefined): void {
  if (value === undefined) {
    Editor.removeMark(editor, key as string);
  } else {
    Editor.addMark(editor, key as string, value);
  }
}

/** Remove all inline formatting marks from the current selection. */
export function removeAllMarks(editor: CustomEditor): void {
  const markKeys: (keyof Omit<CustomText, 'text'>)[] = [
    'bold', 'italic', 'underline', 'strikethrough',
    'color', 'highlight', 'fontFamily', 'fontSize', 'verticalAlign',
  ];
  markKeys.forEach((k) => Editor.removeMark(editor, k as string));
}

// ─────────────────────────────────────────────────────────────────────────────
// Block helpers
// ─────────────────────────────────────────────────────────────────────────────

export function isBlockActive(editor: CustomEditor, type: BlockType): boolean {
  const { selection } = editor;
  if (!selection) return false;

  const [match] = Array.from(
    Editor.nodes(editor, {
      at: Editor.unhangRange(editor, selection),
      match: (n) =>
        !Editor.isEditor(n) &&
        SlateElement.isElement(n) &&
        (n as { type: BlockType }).type === type,
    }),
  );
  return !!match;
}

export function isAlignActive(editor: CustomEditor, align: AlignType): boolean {
  const { selection } = editor;
  if (!selection) return false;

  const [match] = Array.from(
    Editor.nodes(editor, {
      at: Editor.unhangRange(editor, selection),
      match: (n) =>
        !Editor.isEditor(n) &&
        SlateElement.isElement(n) &&
        (n as { align?: AlignType }).align === align,
    }),
  );
  return !!match;
}

/** Toggle a list or non-list block type. */
export function toggleBlock(editor: CustomEditor, type: BlockType): void {
  const isActive = isBlockActive(editor, type);
  const isList = LIST_TYPES.includes(type);

  // Always unwrap existing list wrappers first
  Transforms.unwrapNodes(editor, {
    match: (n) =>
      !Editor.isEditor(n) &&
      SlateElement.isElement(n) &&
      LIST_TYPES.includes((n as { type: BlockType }).type),
    split: true,
  });

  let newType: BlockType;
  if (isActive) {
    newType = 'paragraph'; // turning off → plain paragraph
  } else if (isList) {
    newType = 'list-item'; // items inside the list
  } else {
    newType = type;
  }

  Transforms.setNodes<SlateElement>(editor, { type: newType } as Partial<SlateElement>);

  if (!isActive && isList) {
    Transforms.wrapNodes(editor, { type, children: [] } as SlateElement);
  }
}

/** Set the block type from an HTML tag name (h1 → heading-one, etc.). */
export function setBlockFromTag(editor: CustomEditor, tag: string): void {
  const typeMap: Record<string, BlockType> = {
    p: 'paragraph',
    h1: 'heading-one',
    h2: 'heading-two',
    h3: 'heading-three',
    h4: 'heading-four',
    blockquote: 'blockquote',
    pre: 'code-block',
  };
  const type = typeMap[tag] ?? 'paragraph';

  // Unwrap any list wrappers so we can set plain block types
  Transforms.unwrapNodes(editor, {
    match: (n) =>
      !Editor.isEditor(n) &&
      SlateElement.isElement(n) &&
      LIST_TYPES.includes((n as { type: BlockType }).type),
    split: true,
  });

  Transforms.setNodes<SlateElement>(editor, { type, align: undefined } as Partial<SlateElement>);
}

// ─────────────────────────────────────────────────────────────────────────────
// Alignment
// ─────────────────────────────────────────────────────────────────────────────

export function setAlign(editor: CustomEditor, align: AlignType): void {
  // Toggle off if already active, otherwise set
  const isActive = isAlignActive(editor, align);
  Transforms.setNodes<SlateElement>(
    editor,
    { align: isActive ? undefined : align } as Partial<SlateElement>,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Misc
// ─────────────────────────────────────────────────────────────────────────────

/** Insert a horizontal-rule divider followed by an empty paragraph. */
export function insertDivider(editor: CustomEditor): void {
  Transforms.insertNodes(editor, [
    { type: 'divider', children: [{ text: '' }] } as SlateElement,
    { type: 'paragraph', children: [{ text: '' }] } as SlateElement,
  ]);
}

/** Increase list item depth by wrapping with the same list type. */
export function indentListItem(editor: CustomEditor): void {
  const { selection } = editor;
  if (!selection) return;

  const [listItemMatch] = Array.from(
    Editor.nodes(editor, {
      match: (n) =>
        !Editor.isEditor(n) &&
        SlateElement.isElement(n) &&
        (n as { type: BlockType }).type === 'list-item',
    }),
  );

  if (!listItemMatch) return;

  const [, itemPath] = listItemMatch;
  const parentPath = itemPath.slice(0, -1);
  const [parent] = Editor.node(editor, parentPath);
  const listType: BlockType = SlateElement.isElement(parent)
    ? (parent as { type: BlockType }).type
    : 'bulleted-list';

  Transforms.wrapNodes(editor, { type: listType, children: [] } as SlateElement, {
    at: itemPath,
  });
}

/** Lift a list item up one level (outdent). */
export function outdentListItem(editor: CustomEditor): void {
  const { selection } = editor;
  if (!selection) return;

  Transforms.liftNodes(editor, {
    match: (n) =>
      !Editor.isEditor(n) &&
      SlateElement.isElement(n) &&
      (n as { type: BlockType }).type === 'list-item',
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Active-format snapshot (replaces document.queryCommandState)
// ─────────────────────────────────────────────────────────────────────────────

export interface ActiveFormatsSnapshot {
  bold: boolean;
  italic: boolean;
  underline: boolean;
  strikeThrough: boolean;
  justifyLeft: boolean;
  justifyCenter: boolean;
  justifyRight: boolean;
  justifyFull: boolean;
  insertUnorderedList: boolean;
  insertOrderedList: boolean;
}

export function getActiveFormats(editor: CustomEditor): ActiveFormatsSnapshot {
  const marks = Editor.marks(editor) ?? {};
  return {
    bold: marks.bold === true,
    italic: marks.italic === true,
    underline: marks.underline === true,
    strikeThrough: marks.strikethrough === true,
    justifyLeft: isAlignActive(editor, 'left'),
    justifyCenter: isAlignActive(editor, 'center'),
    justifyRight: isAlignActive(editor, 'right'),
    justifyFull: isAlignActive(editor, 'justify'),
    insertUnorderedList: isBlockActive(editor, 'bulleted-list'),
    insertOrderedList: isBlockActive(editor, 'numbered-list'),
  };
}
