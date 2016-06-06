local Decoder, parent = torch.class('Decoder', 'nn.Module')
include "MaskedLoss.lua"

function Decoder:__init(config, params)
	parent.__init(self)
	self.emb_dim = config.emb_dim
	self.emb_out = config.emb_out
	self.params = params

	self.cell = self:cell()
  self.paramx, self.paramdx = self.cell:getParameters()
  self.paramx:uniform(-self.params.init_weight, self.params.init_weight)
	self.paramdx:zero()

	self.cells = cloneManyTimes(self.cell, self.params.max_nl_length)

	self.s={}
	self.ds={}

	for j = 0, self.params.max_nl_length do
		self.s[j] = {}
		for d = 1, 2 * self.params.layers do
			self.s[j][d] = torch.zeros(self.params.batch_size, self.params.rnn_size):cuda()
		end
	end
	self.cudaones = torch.ones(1):cuda()

	for d = 1, 2 * self.params.layers do
		self.ds[d] = torch.zeros(self.params.batch_size, self.params.rnn_size):cuda()
	end

end

function Decoder:cell()

	  -- decoder with attention cell
	  -- with position input

	  local  all_h = nn.Identity()()
	  local infmask = nn.Identity()()

	  -- inffmask is seq x minibatch


    -- regular decoder cell
	  local x                = nn.Identity()()
 		local y                = nn.Identity()()
		local prev_s           = nn.Identity()()

		local pos              = nn.Identity()()
		local pos_lookup       = nn.LookupTable(101, 50)(pos)

		local i                = {[0] = nn.JoinTable(2)({pos_lookup, nn.LookupTable(self.emb_dim, self.emb_out)(x)}) }

		local next_s           = {}
		local splitted         = {prev_s:split(2 * self.params.layers)}
		for layer_idx = 1, self.params.layers do
			local prev_c         = splitted[2 * layer_idx - 1]
			local prev_h         = splitted[2 * layer_idx]
			local dropped        = nn.Dropout(self.params.dropout)(i[layer_idx - 1])
			local next_c, next_h = lstm(dropped, prev_c, prev_h, 450) -- self.emb_out)
			table.insert(next_s, next_c)
			table.insert(next_s, next_h)
			i[layer_idx] = next_h
		end


		-- ok, now we have the decoder state in i[1]
    -- size of all_h is  minibatch x seq x 400
    -- size of dropped in minibatch x 400 
    -- we want minibatch x seq x 1
    -- Do not use log soft max here.. since we have to take a weighted combination
    -- using these weights
    -- Remember that before doing the softmax, we have to convert those weights
    -- which correspond to the padding to -infinity so that softmax assigns them 
    -- a weight of zero
    --
    local attention_weights = nn.Select(3, 1)(nn.MM()({all_h, nn.Replicate(1, 3)(i[self.params.layers])}))


    local masked_attention_weights = nn.SoftMax()(nn.CAddTable()({attention_weights, nn.Transpose({1,2})(infmask)}))

		-- attention is minibatch x seq --> make this minibatch x seq x 1
		-- all_h is minibatch x seq x 400
		-- we want minibatch x 400 x 1
		-- Then reduce it to minibatch x 400
		
    local context_vector = nn.Select(3, 1)(nn.MM(true, false)({all_h, nn.Replicate(1, 3)(masked_attention_weights)}))


 		local W1 = nn.Linear(self.params.rnn_size, self.params.rnn_size)
 		local W2 = nn.Linear(self.params.rnn_size, self.params.rnn_size)

		-- The final vector. We then softmax this
 		local h_att = nn.Tanh()(nn.CAddTable()({W1(i[self.params.layers]), W2(context_vector)}))

		local dropped          = nn.Dropout(self.params.dropout)(h_att)

 		local h2y              = nn.Linear(self.params.rnn_size, self.emb_dim)
 		local pred             = nn.LogSoftMax()(h2y(dropped))
 		local err              = MaskedLoss()({pred, y})

    return  nn.gModule({x, y, pos, prev_s, all_h, infmask}, {err, nn.Identity()(next_s)}):cuda()
end

function Decoder:forward(state, batch, all_h) 

	for i = 1, (batch.maxY - 1) do
	  err, self.s[i] = unpack(self.cells[i]:forward({batch.y[i], batch.y[i + 1], torch.ones(self.params.batch_size):cuda() * i, self.s[i - 1], all_h, batch.infmask}))

		state.count = state.count + err[2]
		state.normal = state.normal + err[3]
	end
	state.acc = state.count / state.normal
end

function Decoder:backward(batch, all_h)
	for d = 1, #self.ds do
		self.ds[d]:zero()
	end

	local sum_d_all_h = all_h:clone():zero()

	for i = (batch.maxY - 1), 1, -1 do

		local d_prev, d_next, d_s, d_all_h, d_mask
	  d_prev, d_next, d_pos, d_s, d_all_h, d_mask = unpack(self.cells[i]:backward(
			{batch.y[i], batch.y[i + 1], torch.ones(self.params.batch_size):cuda() * i, self.s[i - 1], all_h, batch.infmask},
			{self.cudaones, self.ds}))
 		  sum_d_all_h:add(d_all_h)

		copy_table(self.ds, d_s)
	end

	-- we also need to return the d's that have added up for the encoder
	return sum_d_all_h

end

function Decoder:training()
	g_enable_dropout_all(self.cells, params.dropout)
end

function Decoder:evaluate()
	g_disable_dropout_all(self.cells)
end

function Decoder:updateParameters()
  update_params(self.paramx, self.paramdx, self.params.max_grad_norm, self.params.learningRate, self.params.normalize)
	self.paramdx:zero()
end
