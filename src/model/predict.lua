
include "utils.lua"

function get_predictions(state, ml, bs, enc, dec)

	count = 1

	local ps = {}
	for d = 1, 2 * params.layers do
		ps[d] = torch.zeros(1, params.rnn_size):cuda() -- for prediction
	end

	local predictions = {}
	local alignments = {}

	local start = torch.tic()
	for _, c_batch in pairs(state.batches) do
		local batch = getInCuda(c_batch)


		-- hack. We have to pass a whole batch into encoder. So just replicate.
		batch.x = batch.x:expand(params.max_code_length, params.batch_size)
		batch.mask = batch.mask:expand(params.max_code_length, params.batch_size)
		batch.fmask = batch.fmask:expand(params.max_code_length, params.batch_size)
		batch.infmask = batch.infmask:expand(params.max_code_length, params.batch_size)
		batch.xsizes = batch.xsizes:expand(params.batch_size)


		local all_h = enc:forward(batch.x)
		count = count + 1

		-- now beam search on the second model
		local prediction, aligns = unpack(beam(ps, all_h:narrow(1, 1, 1), batch.infmask:narrow(2, 1, 1), ml, bs, dec))



		if #prediction == 0 then
			table.insert(predictions, {batch.ids[1], 'mysql'})
		else
			table.insert(predictions, {batch.ids[1], table.concat(prediction, ' ')})

			table.insert(prediction, 1, "CODE_START")
			table.insert(prediction, "CODE_END")

			-- Only if alignments are present
			if #aligns ~= 0 then
				alignments[batch.ids[1]] = {}

				-- Add alignment header
				alignmentSQL = {""}
				for i = 1, batch.maxX do
					table.insert(alignmentSQL, vocab.num_to_sql[batch.x[i][1]])
				end

				table.insert(alignments[batch.ids[1]], alignmentSQL)
				for i = 1, math.min(#prediction, ml) do
					local al = {prediction[i]} -- add the NL word
					for j = 1, batch.maxX  do -- add all the alignment scores
						table.insert(al, aligns[i][1][j])
					end
					table.insert(alignments[batch.ids[1]], al) -- insert alignment scores into return value
				end
			end
		end


	end

	return {predictions, alignments}

end

function beam(prevs, all_h, infmask, max_length, beam_size, dec)
	local cnt = 0
	local y = torch.ones(1):cuda()
	local x = torch.ones(1):cuda()

	local default_beam = {prob=0.0,
	str={},
	x=x,
	prevs=prevs,
	alignments={},
	strmap = {},
	pos=1,
}

local beams = {[1]=default_beam}

while cnt < max_length do

	local new_beams = {}

	-- Create the new beams
	for i = 1, #beams do

		if beams[i].x[1] ~= 4 then
			-- Prediction is done except for the first time. So leave it alone

			local tmp = dec:forward({beams[i].x, y, torch.ones(1):cuda() * beams[i].pos, beams[i].prevs, all_h, infmask})[2]


			local fnodes = dec.forwardnodes
			local pred_vector = fnodes[#fnodes].data.mapindex[1].input[1][1]
			local probs, inds = pred_vector:sort()

			-- fetch the alignments.
			local alignments
			--dec:apply(function(m) if tostring(m) == "nn.SoftMax" then alignments = m.output end end)
			-- dec:apply(function(m) if tostring(m) == "cudnn.SoftMax" then alignments = m.output end end)

			for j = 1, beam_size do
				-- negative since its sorted in as
				local p = probs[-1 * j]
				local ind = inds[-1 * j]

				local next_str = table.shallow_copy(beams[i].str)
				local next_map = clone(beams[i].strmap)
				local word = vocab.num_to_nl[tostring(ind)]

				-- BlockRepeatWords
				if word ~= "UNK" and (id == 4 or ind == 3 or #word < 3 or beams[i].strmap[word] == nil) then
					-- we have predicted the last word
					if ind ~= 4 and ind ~= 3 then
						table.insert(next_str, vocab.num_to_nl[tostring(ind)])
						next_map[word] = true
					end

					local next_s = {}
					for d = 1, 2 * params.layers do
						next_s[d] = torch.zeros(1, params.rnn_size):cuda()
					end
					copy_table(next_s, tmp)

					next_beam = {prob=beams[i].prob + p, -- log probabilities
					str=next_str,
					x=torch.ones(1):cuda() * ind,
					prevs=next_s,
					alignments={},
					strmap=next_map,
					pos=beams[i].pos + 1}

					-- Only executed if SoftMax layer corresponding to attention
					-- mechanism is present
					--
					if alignments ~= nil then
						-- Add alignments to new beam
						for k = 1, #(beams[i].alignments) do
							table.insert(next_beam.alignments, beams[i].alignments[k])
						end
						cloned_alignments = alignments:clone()
						table.insert(next_beam.alignments, cloned_alignments)
					end

					table.insert(new_beams, next_beam)
				end

			end
		else
			table.insert(new_beams, beams[i])
		end
	end


	-- prune the beams to beam_size
	table.sort(new_beams, beam_compare)
	-- for k = 1, #new_beams do print(new_beams[k].str .. ' ' .. new_beams[k].prob) end
	for k = beam_size + 1, #new_beams do new_beams[k] = nil end

	beams = new_beams

	cnt = cnt + 1
end

if #beams == 0 then table.insert(beams, default_beam) end -- If there is nothing in the beam, put back default
return {beams[1].str, beams[1].alignments}
end

function beam_compare(a, b)
	-- Adding because we are in log space
	local abp = 0 -- math.min(1 - 10.0/#a.str, 0.0)
	local bbp = 0 --math.min(1 - 10.0/#b.str, 0.0)
	return (a.prob + abp) > (b.prob + bbp)
end

function main()
	local cmd = torch.CmdLine()
	cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed.')
	cmd:option('-encoder',  'None', 'Previously trained encoder')
	cmd:option('-decoder',  'None', 'Previously trained decoder')
	cmd:option('-testfile',  'None', 'Previously trained model')
	cmd:option('-beamsize',  10, 'beam size?')
	cmd:option('-language', 'code', 'Code language')
	cmd:option('-rnn_size', 400, 'Dimension')
	local working_dir = os.getenv("CODENN_WORK")

	cmd:text()
	opt = cmd:parse(arg)

	params =      {
		max_length=20,
		beam_size=opt.beamsize,
		layers=1,
		max_code_length=100,
		max_nl_length=100,
		batch_size=100,
		alignments=opt.alignments,
	  rnn_size=opt.rnn_size}

		print(params)

		init_gpu(opt.gpuidx)

	  vocab = torch.load(working_dir .. '/vocab.data.' .. opt.language)
	  state_test = torch.load(working_dir .. '/eval.data.' .. opt.language)
		encoderCell = torch.load(opt.encoder)
		decoderCell= torch.load(opt.decoder)


		print('predicting')

		g_disable_dropout(encoderCell)
		g_disable_dropout(decoderCell)
		local predictions, alignments  = unpack(get_predictions(state_test, params.max_length, params.beam_size, encoderCell, decoderCell))

		local tmpFilename = os.tmpname()
		local tf = io.open(tmpFilename, 'w')
		for id, aligns in pairs(alignments) do
			tf:write(id.. '\n')
			for _, line in pairs(aligns) do
				tf:write(table.concat(line, "|||||") .. '\n')
			end
		end
		tf:close()
		print('Alignments in  ' .. tmpFilename)

		local tmpFilename = os.tmpname()
		local tf = io.open(tmpFilename, 'w')
		for _, line in ipairs(predictions) do
			tf:write(line[1] .. '\t' .. line[2]  .. '\n')
		end
		tf:close()
		print('Predictions in  ' .. tmpFilename)


	end


	if script_path() == "predict.lua" then
		include "MaskedLoss.lua"
		require('encoder')
		main()
	end
