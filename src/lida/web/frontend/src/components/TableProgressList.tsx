import React from 'react';
import type { TableProgress } from '../hooks/useUploadPipeline';

interface TableProgressListProps {
  tables: TableProgress[];
  dispatchKind: 'single' | 'sqlite' | 'tar' | null;
}

const STATUS_LABEL: Record<TableProgress['status'], string> = {
  pending: 'queued',
  analyzing: 'summarizing',
  done: 'analyzed',
  error: 'error',
};

export const TableProgressList: React.FC<TableProgressListProps> = ({ tables, dispatchKind }) => {
  if (tables.length === 0) return null;

  const heading =
    dispatchKind === 'sqlite' ? 'SQLite tables'
    : dispatchKind === 'tar' ? 'Archive entries'
    : 'Dataset';

  return (
    <div className="table-progress">
      <div className="table-progress__heading">{heading}</div>
      <ul className="table-progress__list">
        {tables.map((t) => (
          <li key={t.name} className={`table-progress__item table-progress__item--${t.status}`}>
            <div className="table-progress__row">
              <span className="table-progress__name" title={t.name}>{t.name}</span>
              <span className="table-progress__status">{STATUS_LABEL[t.status]}</span>
            </div>
            {t.error && <div className="table-progress__error">{t.error}</div>}
          </li>
        ))}
      </ul>
    </div>
  );
};
