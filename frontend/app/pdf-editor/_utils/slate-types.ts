import type { BaseEditor, Descendant } from 'slate';
import type { ReactEditor } from 'slate-react';
import type { HistoryEditor } from 'slate-history';

export type BlockType =
  | 'paragraph'
  | 'heading-one'
  | 'heading-two'
  | 'heading-three'
  | 'heading-four'
  | 'blockquote'
  | 'code-block'
  | 'bulleted-list'
  | 'numbered-list'
  | 'list-item'
  | 'divider'
  /** Opaque void block — stores raw HTML for tables, figures, charts, etc. */
  | 'raw-html';

export type AlignType = 'left' | 'center' | 'right' | 'justify';

export type CustomElement = {
  type: BlockType;
  align?: AlignType;
  /** Raw HTML payload — only used when type === 'raw-html' */
  rawHtml?: string;
  children: Descendant[];
};

export type CustomText = {
  text: string;
  bold?: true;
  italic?: true;
  underline?: true;
  strikethrough?: true;
  /** Font-family string, e.g. "Georgia, serif" */
  fontFamily?: string;
  /** Numeric string in pt, e.g. "12" */
  fontSize?: string;
  color?: string;
  highlight?: string;
  verticalAlign?: 'super' | 'sub';
};

export type CustomEditor = BaseEditor & ReactEditor & HistoryEditor;

declare module 'slate' {
  interface CustomTypes {
    Editor: CustomEditor;
    Element: CustomElement;
    Text: CustomText;
  }
}
