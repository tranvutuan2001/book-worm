import { FONTS, FONT_SIZES } from '@/config/pdf-editor-font';
import { ToolbarButton, ToolbarDivider } from './ToolbarButton';

export interface ActiveFormats {
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

interface EditorToolbarProps {
  activeFormats: ActiveFormats;
  fontFamily: string;
  fontSize: string;
  onExec: (command: string, value?: string) => void;
  onBlockFormat: (tag: string) => void;
  onFontFamilyChange: (family: string) => void;
  onFontSizeChange: (size: string) => void;
}

export default function EditorToolbar({
  activeFormats,
  fontFamily,
  fontSize,
  onExec,
  onBlockFormat,
  onFontFamilyChange,
  onFontSizeChange,
}: EditorToolbarProps) {
  return (
    <div className="bg-white border-b border-gray-200 px-4 py-2 flex flex-wrap items-center gap-1 shrink-0 shadow-sm">
      {/* ── Undo / Redo ── */}
      <ToolbarButton onClick={() => onExec('undo')} title="Undo">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
        </svg>
      </ToolbarButton>
      <ToolbarButton onClick={() => onExec('redo')} title="Redo">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M21 10H11a8 8 0 00-8 8v2m18-10l-6 6m6-6l-6-6" />
        </svg>
      </ToolbarButton>

      <ToolbarDivider />

      {/* ── Block / paragraph style ── */}
      <select
        className="text-xs border border-gray-200 rounded px-2 py-1 text-gray-700 focus:outline-none focus:ring-1 focus:ring-purple-400"
        onChange={(e) => onBlockFormat(e.target.value)}
        defaultValue="p"
        title="Paragraph style"
      >
        <option value="p">Paragraph</option>
        <option value="h1">Heading 1</option>
        <option value="h2">Heading 2</option>
        <option value="h3">Heading 3</option>
        <option value="h4">Heading 4</option>
        <option value="blockquote">Quote</option>
        <option value="pre">Code block</option>
      </select>

      <ToolbarDivider />

      {/* ── Font family ── */}
      <select
        className="text-xs border border-gray-200 rounded px-2 py-1 text-gray-700 focus:outline-none focus:ring-1 focus:ring-purple-400 max-w-[140px]"
        value={fontFamily}
        onChange={(e) => onFontFamilyChange(e.target.value)}
        title="Font family"
      >
        {FONTS.map((f) => (
          <option key={f} value={f} style={{ fontFamily: f }}>
            {f.split(',')[0]}
          </option>
        ))}
      </select>

      {/* ── Font size ── */}
      <select
        className="text-xs border border-gray-200 rounded px-2 py-1 text-gray-700 focus:outline-none focus:ring-1 focus:ring-purple-400 w-16"
        value={fontSize}
        onChange={(e) => onFontSizeChange(e.target.value)}
        title="Font size (pt)"
      >
        {FONT_SIZES.map((s) => (
          <option key={s} value={s}>{s}pt</option>
        ))}
      </select>

      <ToolbarDivider />

      {/* ── Text formatting ── */}
      <ToolbarButton onClick={() => onExec('bold')} active={activeFormats.bold} title="Bold (Ctrl+B)">
        <strong className="text-xs">B</strong>
      </ToolbarButton>
      <ToolbarButton onClick={() => onExec('italic')} active={activeFormats.italic} title="Italic (Ctrl+I)">
        <em className="text-xs font-serif">I</em>
      </ToolbarButton>
      <ToolbarButton onClick={() => onExec('underline')} active={activeFormats.underline} title="Underline (Ctrl+U)">
        <span className="text-xs underline">U</span>
      </ToolbarButton>
      <ToolbarButton onClick={() => onExec('strikeThrough')} active={activeFormats.strikeThrough} title="Strikethrough">
        <span className="text-xs line-through">S</span>
      </ToolbarButton>

      <ToolbarDivider />

      {/* ── Alignment ── */}
      <ToolbarButton onClick={() => onExec('justifyLeft')} active={activeFormats.justifyLeft} title="Align left">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h10M4 18h13" />
        </svg>
      </ToolbarButton>
      <ToolbarButton onClick={() => onExec('justifyCenter')} active={activeFormats.justifyCenter} title="Center">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M7 12h10M6 18h12" />
        </svg>
      </ToolbarButton>
      <ToolbarButton onClick={() => onExec('justifyRight')} active={activeFormats.justifyRight} title="Align right">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M10 12h10M7 18h13" />
        </svg>
      </ToolbarButton>
      <ToolbarButton onClick={() => onExec('justifyFull')} active={activeFormats.justifyFull} title="Justify">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </ToolbarButton>

      <ToolbarDivider />

      {/* ── Lists ── */}
      <ToolbarButton onClick={() => onExec('insertUnorderedList')} active={activeFormats.insertUnorderedList} title="Bullet list">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
        </svg>
      </ToolbarButton>
      <ToolbarButton onClick={() => onExec('insertOrderedList')} active={activeFormats.insertOrderedList} title="Numbered list">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
      </ToolbarButton>

      <ToolbarDivider />

      {/* ── Indent / Outdent ── */}
      <ToolbarButton onClick={() => onExec('indent')} title="Indent">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M4 6h16M4 12h8m-8 6h16M9 9l3 3-3 3" />
        </svg>
      </ToolbarButton>
      <ToolbarButton onClick={() => onExec('outdent')} title="Outdent">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M4 6h16M4 12h8m-8 6h16M12 9l-3 3 3 3" />
        </svg>
      </ToolbarButton>

      <ToolbarDivider />

      {/* ── Misc ── */}
      <ToolbarButton onClick={() => onExec('insertHorizontalRule')} title="Horizontal rule">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 12h16" />
        </svg>
      </ToolbarButton>
      <ToolbarButton onClick={() => onExec('removeFormat')} title="Clear formatting">
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </ToolbarButton>
    </div>
  );
}
