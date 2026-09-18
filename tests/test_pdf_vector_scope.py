import hashlib
import unittest
from unittest.mock import patch

import mr_corpus
import mr_index
import mr_tools


def corpus():
    rows=[]
    for i,(source,score) in enumerate([('stock',.99),('pdf',.6),('case',.9),('pdf',.8),('pdf',.7)]):
        rows.append({'source':source,'file':f'{i}.md','heading':f'p{i}' if source=='pdf' else 'body',
                     'text':'fixture text','id':str(i),'md5':hashlib.md5(str(i).encode()).hexdigest(),
                     'purpose':'knowledge','fixture_score':score})
    return rows


class FakeVector:
    constructed=[]
    def __init__(self,chunks,model=None,spec=None):
        self.chunks=list(chunks)
        self.spec=spec
        self._model=model if model is not None else object()
        self.available,self.err=True,''
        self.constructed.append(self)
    def search(self,query,topk=50,mask=None):
        candidates=range(len(self.chunks)) if mask is None else mask
        return sorted(((i,self.chunks[i]['fixture_score']) for i in candidates),key=lambda x:x[1],reverse=True)[:topk]


class PdfVectorScopeTests(unittest.TestCase):
    def setUp(self):
        self.chunks=corpus()
        FakeVector.constructed=[]
        self.enterContext(patch.object(mr_index,'VecIndex',FakeVector))
        self.enterContext(patch.object(mr_tools,'_STATE',{}))
        self.enterContext(patch.object(mr_tools,'LOG_ON',False))
        self.enterContext(patch.object(mr_corpus,'signature',return_value='frozen'))
        self.enterContext(patch.object(mr_corpus,'load_chunks',side_effect=lambda:self.chunks))

    def test_subset_matches_full_masked_scores_and_returns_global_indices(self):
        spec=mr_index.model_spec('bge-m3')
        full=FakeVector(self.chunks,spec=spec)
        subset=mr_tools._SubVec(self.chunks,[1,3,4],spec=spec)
        self.assertEqual(subset.search('q',mask={1,4}),full.search('q',mask={1,4}))
        self.assertEqual(subset.search('q',topk=2),full.search('q',topk=2,mask={1,3,4}))

    def test_real_pdf_route_never_embeds_stock_or_note_rows(self):
        result=mr_tools.memory_search('fixture',source='pdf',mode='vec',model='bge-m3',rerank=False)
        self.assertTrue(result['ok'])
        self.assertEqual(result['mode'],'vec')
        self.assertEqual([h['file'] for h in result['hits']],['3.md','4.md','1.md'])
        self.assertEqual({c['source'] for v in FakeVector.constructed for c in v.chunks},{'pdf'})

    def test_pdf_and_full_search_cache_keys_are_distinct_but_weights_shared(self):
        pdf=mr_tools._ensure('vec',model='bge-m3',vec_scope='pdf')['vec']
        full=mr_tools._ensure('vec',model='bge-m3')['vec']
        self.assertIsNot(pdf,full)
        self.assertIs(pdf._model,full._model)
        self.assertEqual(len(pdf.idx),3)
        self.assertEqual(len(full.chunks),5)

    def test_reload_preserves_pdf_scope_and_remaps_global_positions(self):
        before=mr_tools._ensure('vec',model='bge-m3',vec_scope='pdf')['vec']
        old_weights=before._model
        self.chunks=[dict(self.chunks[0],file='new-stock.md')]+self.chunks
        mr_tools._rebuild('fixture reload')
        after=mr_tools._ensure('vec',model='bge-m3',vec_scope='pdf')['vec']
        self.assertEqual(after.idx,[2,4,5])
        self.assertEqual(after.search('q')[0][0],4)
        self.assertIs(after._model,old_weights)
        self.assertTrue(all(c['source']=='pdf' for v in FakeVector.constructed for c in v.chunks))


if __name__=='__main__':
    unittest.main()
