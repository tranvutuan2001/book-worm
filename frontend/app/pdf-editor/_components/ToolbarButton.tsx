export interface ToolbarButtonProps {
  onClick: () => void;
  active?: boolean;
  title: string;
  children: React.ReactNode;
  disabled?: boolean;
}

export function ToolbarButton({ onClick, active, title, children, disabled }: ToolbarButtonProps) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onMouseDown={(e) => {
        e.preventDefault(); // prevent the editor from losing focus
        onClick();
      }}
      className={`p-1.5 rounded text-sm leading-none transition-all ${
        active
          ? 'bg-purple-600 text-white shadow-sm'
          : 'text-gray-700 hover:bg-purple-100 hover:text-purple-700'
      } disabled:opacity-40 disabled:cursor-not-allowed`}
    >
      {children}
    </button>
  );
}

export function ToolbarDivider() {
  return <span className="w-px h-5 bg-gray-300 mx-1 shrink-0" />;
}
