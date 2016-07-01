
include "utils.lua"
local debugger = require("fb.debugger")
include "MaskedLoss.lua"
require('encoder')
py = require('fb.python')
require('buildData')

server = {}
function get_predictions(xbatch, ybatch, enc, dec)

	local ps = {}
	for d = 1, 2 * params.layers do
		ps[d] = torch.zeros(1, params.rnn_size):cuda() -- for prediction
	end

	xbatch.x = xbatch.x:expand(params.max_code_length, params.batch_size)
	xbatch.mask = xbatch.mask:expand(params.max_code_length, params.batch_size)
	xbatch.fmask = xbatch.fmask:expand(params.max_code_length, params.batch_size)
	xbatch.infmask = xbatch.infmask:expand(params.max_code_length, params.batch_size)
	xbatch.xsizes = xbatch.xsizes:expand(params.batch_size)

	local all_h = enc:forward(xbatch.x)

	return computeProb(ybatch, ps, all_h:narrow(1, 1, 1), xbatch.infmask:narrow(2, 1, 1), dec)

end

function computeProb(batch, prevs, all_h, infmask, dec)
	local y = torch.ones(1):cuda()

	local i = 1
	local prob = 0
	repeat
		local tmp = dec:forward({batch.y[i], y, torch.ones(1):cuda() * i, prevs, all_h, infmask})[2]
		local fnodes = dec.forwardnodes
		local pred_vector = fnodes[#fnodes].data.mapindex[1].input[1][1]
		prob = prob + pred_vector[batch.y[i + 1][1]]
		copy_table(prevs, tmp)
		i = i + 1
	until i > params.max_nl_length or batch.y[i][1] == 1

	return prob
end

function dist_compare(a, b)
	return a.prob > b.prob
end

function server.init(port)
    state = "running"
	require "socket"
	
	-- create a TCP socket and bind it to the local host, at any port
	server = assert(socket.bind("*", port))
	-- find out which port the OS chose for us
	ip, port = server:getsockname()
end

function initPythonMethods()
	py.exec([=[
from model.buildData import get_data_map
	]=])
end

function main()

	local cmd = torch.CmdLine()
	cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed.')
	cmd:option('-encoder',  'None', 'Previously trained encoder')
	cmd:option('-decoder',  'None', 'Previously trained decoder')
	cmd:option('-rnn_size', 400, 'Dimension')
	cmd:option('-language', 'code', 'Code language')
	cmd:option('-port', 9090, 'Server port')
	cmd:option('-max_nl_length', 100, 'length')
	cmd:option('-max_code_length', 100, 'length')

	local working_dir = os.getenv("CODENN_WORK")

	cmd:text()
	opt = cmd:parse(arg)

	params =      {
		max_length=20,
		layers=1,
		max_code_length=opt.max_code_length,
		max_nl_length=opt.max_nl_length,
		batch_size=100,
	  rnn_size=opt.rnn_size
	}

	init_gpu(opt.gpuidx)
	initPythonMethods()

	-- preload vocab and models
	vocab = torch.load(working_dir .. '/vocab.data.' .. opt.language)
	print("Vocabulary loaded")
  encoderCell = torch.load(opt.encoder)
  decoderCell= torch.load(opt.decoder)

	state_train = torch.load(working_dir .. '/train_batch_1.data.' .. opt.language)
	state_train.name = "training"

	print("Models loaded")

  g_disable_dropout(encoderCell)
	g_disable_dropout(decoderCell)

  server.init(opt.port)

	while state == "running" do

		-- wait for a connection from any client
		print("wait for connection on port: " .. opt.port)
		local client = server:accept()
		-- make sure we don't block waiting for this client's line
		--client:settimeout(10)

		while true do
			-- tickrate timer start
			-- receive the line
			nl, err = client:receive()
			print(nl)
			if (nl== "exit") then
				break
			end
      -- TEST: nl = "get me sorting code"
			-- NOTE: line should not have any tabs
			local filename = '/tmp/nl2code.tmpfile'
			local f = io.open(filename, 'w')
			f:write('23\t23\t' .. nl .. '\t' .. 'code()' .. '\t' .. '0.45')
			f:close()

	    local py_data = py.eval('get_data_map(f, vocab, dskip, msql, mnl)', {f=filename, vocab=vocab, dskip=true, msql=params.max_code_length, mnl=params.max_nl_length})
      local data = get_data_map(py_data, vocab, 1, true)

		  local n_batch = getInCuda(data.batches[1])
		  local ranks = {}

			-- Now we need to convert 
			for i = 1, #state_train.batches do
				local batch = state_train.batches[i]
				local c_batch = getInCuda(batch)
			  local prob = get_predictions(c_batch, n_batch, encoderCell, decoderCell)
			  table.insert(ranks, {p=prob, c=batch.code})
			end

	    table.sort(ranks, function (a,b) 
	    	return a.p > b.p
			end)

			client:send(tostring(ranks[1].c[1]) .. "\n")
		end

		-- done with client, close the object
		client:close()
	end

end



if script_path() == "nl2code.lua" then
	main()
end
