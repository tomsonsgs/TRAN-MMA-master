# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 00:56:18 2019

@author: tang
"""
seed=102
vocab="vocab.bin"
train_file="train.bin"
dropout=0.3
hidden_size=256
embed_size=100
action_embed_size=100
field_embed_size=32
type_embed_size=32
lr_decay=0.5
beam_size=5
patience=2
lstm='lstm'
col_att='affine'
model_name='wiki'
def updatetest(opt):
    model_name1='wikitest.decode1'
#

    opt.cuda =True
    opt.mode ='test'
    opt.load_model='saved_models/wikisql_bk/'+model_name+'.bin'
    opt.beam_size=5 
    opt.parser='wikisql_parser'
    opt.evaluator='wikisql_evaluator'
    opt.sql_db_file='data/wikisql1/test.db'
    opt.test_file='data/wikisql1/test.bin'
    opt.save_decode_to='decodes/wikisql/'+model_name1
    opt.decode_max_time_step=50
def update(opt):
    opt.cuda=True
    opt.seed=seed
    opt.mode='train'
    opt.batch_size=16
    opt.parser='wikisql_parser' 
    opt.asdl_file='asdl/lang/sql/sql_asdl.txt'
    opt.transition_system='sql' 
    opt.evaluator='wikisql_evaluator' 
    opt.train_file='data/wikisql1/'+train_file
    opt.dev_file='data/wikisql1/test.bin' 
    opt.sql_db_file='data/wikisql1/test.db' 
    opt.vocab='data/wikisql1/'+vocab
    opt.glove_embed_path='data/contrib/glove.6B.100d.txt'
    opt.lstm =lstm
    opt.column_att =col_att
    opt.no_parent_state =True
    opt.no_parent_field_embed =True
    opt.no_parent_field_type_embed =True
    opt.no_parent_production_embed =True
    opt.hidden_size =hidden_size
    opt.embed_size =embed_size
    opt.action_embed_size =action_embed_size
    opt.field_embed_size =field_embed_size
    opt.type_embed_size =type_embed_size
    opt.dropout =dropout
    opt.patience =patience
    opt.max_num_trial =5 
    opt.lr_decay =lr_decay
    opt.glorot_init=True
    opt.beam_size =beam_size
    opt.eval_top_pred_only =True
    opt.decode_max_time_step=50
    opt.log_every=500
    opt.save_to='saved_models/wikisql_bk/'+model_name
#python -u exp.py \
#    --cuda \
#    --seed ${seed} \
#    --mode train \
#    --batch_size 64 \
#    --parser wikisql_parser \
#    --asdl_file asdl/lang/sql/sql_asdl.txt \
#    --transition_system sql \
#    --evaluator wikisql_evaluator \
#    --train_file data/wikisql/${train_file} \
#    --dev_file data/wikisql/dev.bin \
#    --sql_db_file data/wikisql/dev.db \
#    --vocab data/wikisql/${vocab} \
#    --glove_embed_path data/contrib/glove.6B.100d.txt \
#    --lstm ${lstm} \
#    --column_att ${col_att} \
#    --no_parent_state \
#    --no_parent_field_embed \
#    --no_parent_field_type_embed \
#    --no_parent_production_embed \
#    --hidden_size ${hidden_size} \
#    --embed_size ${embed_size} \
#    --action_embed_size ${action_embed_size} \
#    --field_embed_size ${field_embed_size} \
#    --type_embed_size ${type_embed_size} \
#    --dropout ${dropout} \
#    --patience ${patience} \
#    --max_num_trial 5 \
#    --lr_decay ${lr_decay} \
#    --glorot_init \
#    --beam_size ${beam_size} \
#    --eval_top_pred_only \
#    --decode_max_time_step 50 \
#    --log_every 10 \
#    --save_to saved_models/wikisql/${model_name}
